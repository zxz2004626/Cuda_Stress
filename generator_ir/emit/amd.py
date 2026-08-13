"""HIP backend: render a Program IR to a self-contained .hip file."""

from __future__ import annotations

from generator_ir.ir.nodes import WarpShuffle
from generator_ir.ir.program import Func, Program, LaunchConfig
from generator_ir.emit.base_emitter import SourceWriter, write_safe_macros, write_diverge_macros
from generator_ir.emit.nvidia import CudaEmitter
from generator_ir.emit.host_harness import HostHarness, HIP_API


class HipEmitter(CudaEmitter):
    """Program -> HIP source (shares the CUDA structure; differs in API names,
    a few type spellings, and maskless warp shuffles)."""

    def bfloat16_name(self) -> str:
        return "hip_bfloat16"

    def _float_to_bf16(self) -> str:
        return "__float2bfloat16"

    def _call(self, c) -> str:
        """HIP overload resolution is stricter: __ffs/__ffsll need exact
        argument types (an unsigned long long temporary is ambiguous)."""
        if c.fn in ("__ffs", "__clz", "__popc",
                    "__ffsll", "__clzll", "__popcll") and c.args:
            inner = self.expr(c.args[0])
            if c.fn in ("__ffs", "__ffsll"):
                cast_ty = "long long" if c.fn == "__ffsll" else "int"
            else:
                cast_ty = ("unsigned long long" if c.fn.endswith("ll")
                           else "unsigned int")
            return f"{c.fn}(({cast_ty})({inner}))"
        return super()._call(c)

    def _warp_vote(self, wv) -> str:
        """HIP vote intrinsics require a 64-bit mask."""
        mask = self.expr(wv.mask)
        pred = self.expr(wv.predicate)
        return f"__{wv.kind}((unsigned long long)({mask}), {pred})"

    def _warp_shuffle(self, ws: WarpShuffle) -> str:
        """HIP: maskless shuffle forms (AMD wavefronts execute lockstep; the
        mask argument is unnecessary and the maskless forms work on both ROCm
        6.x and 7.x)."""
        _MASKLESS = {
            "shfl_sync": "__shfl",
            "shfl_down_sync": "__shfl_down",
            "shfl_up_sync": "__shfl_up",
            "shfl_xor_sync": "__shfl_xor",
        }
        name = _MASKLESS.get(ws.kind, f"__{ws.kind}")
        val = self.expr(ws.value)
        delta = self.expr(ws.delta)
        if ws.width != 32:
            return f"{name}({val}, {delta}, {ws.width})"
        return f"{name}({val}, {delta})"

    def rendersrc(self, program: Program) -> str:
        self.w = SourceWriter()
        self._program = program
        cfg = program.launch

        self._headers(program, cfg)
        self.w.blank()

        from generator_ir.ir.statements import Declare
        from generator_ir.ir.program import StructDecl
        for d in program.decls:
            if isinstance(d, StructDecl):
                self.struct(d)
                self.w.blank()
        for d in program.decls:
            if isinstance(d, Declare):
                self.stmt(d)
        if any(isinstance(d, Declare) for d in program.decls):
            self.w.blank()

        if cfg.needs_constant_mem:
            self.w.line("__constant__ unsigned int _cmem[256];")
            self.w.blank()

        for d in program.decls:
            if isinstance(d, Func) and not d.kx_is_kernel():
                self.function(d)
                self.w.blank()

        kernel = program.fetch_kernel()
        if kernel is not None:
            if cfg.time_clock64:
                self._kernel_with_clock64(kernel)
            else:
                self.function(kernel)
            self.w.blank()
            HostHarness(self.w, HIP_API, cfg,
                        type_name=self.type_name).emit(kernel, program)

        return self.w.text()

    def _headers(self, program: Program, cfg: LaunchConfig) -> None:
        w = self.w
        from generator_ir.ir.typedecl import THalf, TBF16
        w.raw("#include <stdio.h>")
        w.raw("#include <stdlib.h>")
        w.raw("#include <stdint.h>")
        w.raw("#include <string.h>")
        w.raw("#include <math.h>")
        w.raw("#include <hip/hip_runtime.h>")
        if self._uses_type(program, THalf):
            w.raw("#include <hip/hip_fp16.h>")
        if self._uses_type(program, TBF16):
            w.raw("#include <hip/hip_bfloat16.h>")
        coop_included = False
        if self._uses_rawtext(program, "cooperative_groups"):
            w.raw("#include <hip/hip_cooperative_groups.h>")
            coop_included = True
        # CUDA header spellings remap to their HIP equivalents (None = drop).
        _HIP_HEADER_MAP = {
            "<cuda_fp16.h>": "<hip/hip_fp16.h>",
            "<cuda_bf16.h>": "<hip/hip_bfloat16.h>",
            "<cuda_runtime.h>": "<hip/hip_runtime.h>",
            "<cooperative_groups.h>": "<hip/hip_cooperative_groups.h>",
            # HIP cooperative_groups includes memcpy_async support
            # in the main header; no separate include needed.
            "<cooperative_groups/memcpy_async.h>": None,
        }
        for h in sorted(cfg.extra_headers):
            mapped = _HIP_HEADER_MAP.get(h, h)
            if mapped is None or mapped == "<hip/hip_runtime.h>":
                continue
            if mapped == "<hip/hip_cooperative_groups.h>" and coop_included:
                continue
            w.raw(f"#include {mapped}")
        w.blank()
        write_safe_macros(w)
        write_diverge_macros(w)
        w.raw("#define GPU_HIP_CHECK(call) do { "
          "hipError_t _gpu_err = (call); "
          "if (_gpu_err != hipSuccess) { "
          "fprintf(stderr, \"HIP_ERROR %s:%d: %s: %s\\n\", "
          "__FILE__, __LINE__, #call, hipGetErrorString(_gpu_err)); "
          "return 2; "
          "} "
          "} while (0)")


def translate_hip(program: Program) -> str:
    return HipEmitter().rendersrc(program)
