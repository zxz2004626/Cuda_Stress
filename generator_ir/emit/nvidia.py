"""CUDA backend: render a Program IR to a self-contained .cu file."""

from __future__ import annotations

from generator_ir.ir.typedecl import (
    GType, TArray, THalf, TBF16, TPointer, TStruct, TVec, TQualified,
)
from generator_ir.ir.statements import Declare, RawCode
from generator_ir.ir.program import Func, StructDecl, Program, LaunchConfig
from generator_ir.emit.base_emitter import (
    BaseEmitter, SourceWriter, write_safe_macros, write_diverge_macros,
)
from generator_ir.emit.host_harness import HostHarness, CUDA_API
from generator_ir.emit.timing_modes import emit_clock64_kernel_prefix, emit_clock64_kernel_suffix


class CudaEmitter(BaseEmitter):
    """Program -> CUDA source."""

    def rendersrc(self, program: Program) -> str:
        self.w = SourceWriter()
        self._program = program
        cfg = program.launch

        self._headers(program, cfg)
        self.w.blank()

        for d in program.decls:
            if isinstance(d, StructDecl):
                self.struct(d)
                self.w.blank()

        top_level_decls = [d for d in program.decls if isinstance(d, Declare)]
        if top_level_decls:
            # Stable unmangled names let the Driver API initialize module
            # globals with cuModuleGetGlobal.
            self.w.line('extern "C" {')
            self.w.kx_indent()
            for d in top_level_decls:
                self.stmt(d)
            self.w.kx_dedent()
            self.w.line("}")
        if any(isinstance(d, Declare) for d in program.decls):
            self.w.blank()

        if cfg.needs_constant_mem:
            self.w.line('extern "C" __constant__ unsigned int _cmem[256];')
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
            HostHarness(self.w, CUDA_API, cfg,
                        type_name=self.type_name).emit(kernel, program)

        return self.w.text()

    def signature(self, f: Func) -> str:
        sig = super().signature(f)
        if f.kx_is_kernel():
            return 'extern "C" ' + sig
        return sig

    # -- headers -------------------------------------------------------------

    def _headers(self, program: Program, cfg: LaunchConfig) -> None:
        w = self.w
        w.raw("#include <stdio.h>")
        w.raw("#include <stdlib.h>")
        w.raw("#include <stdint.h>")
        w.raw("#include <string.h>")
        w.raw("#include <math.h>")
        w.raw("#include <cuda_runtime.h>")
        if self._uses_type(program, THalf):
            w.raw("#include <cuda_fp16.h>")
        if self._uses_type(program, TBF16):
            w.raw("#include <cuda_bf16.h>")
        if self._uses_function(program, {"__pipeline_commit", "__pipeline_wait_prior"}):
            w.raw("#include <cuda_pipeline_primitives.h>")
        coop = False
        if self._uses_rawtext(program, "cuda::barrier"):
            w.raw("#include <cuda/barrier>")
            w.raw("#include <cooperative_groups.h>")
            coop = True
        if not coop and self._uses_rawtext(program, "cooperative_groups"):
            w.raw("#include <cooperative_groups.h>")
            coop = True
        for h in sorted(cfg.extra_headers):
            w.raw(f"#include {h}")
        w.blank()
        write_safe_macros(w)
        write_diverge_macros(w)
        w.raw("#define GPU_CUDA_CHECK(call) do { "
          "cudaError_t _gpu_err = (call); "
          "if (_gpu_err != cudaSuccess) { "
          "fprintf(stderr, \"CUDA_ERROR %s:%d: %s: %s\\n\", "
          "__FILE__, __LINE__, #call, cudaGetErrorString(_gpu_err)); "
          "return 2; "
          "} "
          "} while (0)")

    # -- clock64 kernel wrapper ------------------------------------------------

    def _kernel_with_clock64(self, kernel: Func) -> None:
        self.w.line("__device__ long long d_clock_result = 0;")
        self.w.blank()
        self.w.line(f"{self.signature(kernel)} {{")
        self.w.kx_indent()
        emit_clock64_kernel_prefix(self.w)
        if kernel.body:
            self._stmts(kernel.body)
        emit_clock64_kernel_suffix(self.w)
        self.w.kx_dedent()
        self.w.line("}")

    # -- usage scans -------------------------------------------------------------

    @staticmethod
    def _walk_objs(program: Program):
        from generator_ir.ir.traversal import traverse
        yield from traverse(program)

    @staticmethod
    def _type_contains(t, ty) -> bool:
        """Recursively unwrap composite types looking for *ty*.

        Covers TVec.elem, TArray.elem, TPointer.base, TQualified.base, and
        TStruct field types (so a half-precision type nested inside a vector,
        qualified wrapper, pointer/array, or struct field is detected)."""
        if not isinstance(t, GType):
            return False
        if isinstance(t, ty):
            return True
        for attr in ("elem", "base"):
            child = getattr(t, attr, None)
            if isinstance(child, GType) and CudaEmitter._type_contains(child, ty):
                return True
        if isinstance(t, TStruct):
            for _fname, ft in t.fields:
                if CudaEmitter._type_contains(ft, ty):
                    return True
        return False

    def _uses_type(self, program: Program, ty) -> bool:
        for node in self._walk_objs(program):
            t = getattr(node, "ty", None) or getattr(node, "gtype", None)
            if t is not None and self._type_contains(t, ty):
                return True
            # StructDecl carries its TStruct on .struct; the generic traverse
            # does not descend into types, so check the field types here.
            if isinstance(node, StructDecl) and self._type_contains(
                    node.struct, ty):
                return True
        return False

    def _uses_function(self, program: Program, names: set) -> bool:
        from generator_ir.ir.nodes import Call
        for node in self._walk_objs(program):
            if isinstance(node, Call) and node.fn in names:
                return True
        return False

    def _uses_rawtext(self, program: Program, needle: str) -> bool:
        for node in self._walk_objs(program):
            if isinstance(node, RawCode) and needle in node.text:
                return True
        return False


def translate_cuda(program: Program) -> str:
    return CudaEmitter().rendersrc(program)
