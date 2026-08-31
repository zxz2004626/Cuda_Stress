"""Host-side benchmark harness generation.

One harness generator shared by the CUDA and HIP backends, parameterized by a
runtime-API table.  The emitted program:

  1. allocates + fills the input buffer with a deterministic pattern,
  2. runs the kernel under one of five timing modes,
  3. re-runs once into a fresh output buffer and prints a host-side checksum.

Output contract (parsed by generator_ir.run):
  simple:    avg_time_ms=%.3f
  ci:        KERNEL_TIME_MEAN_US: %.6f  (+ CI bounds, margin, batch, samples)
  clock64:   CLOCK64_MEDIAN_CYCLES %lld (+ trimmed mean)
  loop:      avg_time_ms=%.6f + LOOP_N: %d
  profiler:  (no timing; an external profiler measures)
  all:       checksum=%llu
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from generator_ir.ir.typedecl import (
    GType, TPointer, TFloat, TDouble, THalf, TBF16, TBool, TChar, TSChar,
    TShort, TUShort, TInt, TUInt, TLong, TULong,
)
from generator_ir.ir.program import Func, LaunchConfig
from generator_ir.emit.base_emitter import SourceWriter


@dataclass(frozen=True)
class RuntimeApi:
    """Names of the backend runtime entry points used by the harness."""
    error_check_call: str      # macro name for checked calls (defined by emitter)
    malloc: str
    free: str
    memcpy: str
    memset: str
    memcpy_h2d: str
    memcpy_d2h: str
    memcpy_to_symbol: str
    memcpy_from_symbol: str
    event_t: str
    event_create: str
    event_destroy: str
    event_record: str
    event_sync: str
    event_elapsed: str
    device_sync: str
    get_last_error: str


CUDA_API = RuntimeApi(
    error_check_call="GPU_CUDA_CHECK",
    malloc="cudaMalloc", free="cudaFree",
    memcpy="cudaMemcpy", memset="cudaMemset",
    memcpy_h2d="cudaMemcpyHostToDevice", memcpy_d2h="cudaMemcpyDeviceToHost",
    memcpy_to_symbol="cudaMemcpyToSymbol",
    memcpy_from_symbol="cudaMemcpyFromSymbol",
    event_t="cudaEvent_t", event_create="cudaEventCreate",
    event_destroy="cudaEventDestroy", event_record="cudaEventRecord",
    event_sync="cudaEventSynchronize", event_elapsed="cudaEventElapsedTime",
    device_sync="cudaDeviceSynchronize", get_last_error="cudaGetLastError",
)

HIP_API = RuntimeApi(
    error_check_call="GPU_HIP_CHECK",
    malloc="hipMalloc", free="hipFree",
    memcpy="hipMemcpy", memset="hipMemset",
    memcpy_h2d="hipMemcpyHostToDevice", memcpy_d2h="hipMemcpyDeviceToHost",
    memcpy_to_symbol="hipMemcpyToSymbol",
    memcpy_from_symbol="hipMemcpyFromSymbol",
    event_t="hipEvent_t", event_create="hipEventCreate",
    event_destroy="hipEventDestroy", event_record="hipEventRecord",
    event_sync="hipEventSynchronize", event_elapsed="hipEventElapsedTime",
    device_sync="hipDeviceSynchronize", get_last_error="hipGetLastError",
)


# Deterministic host input patterns (no rand(); identical for every binary).
_INPUT_INIT = {
    "hash":        "h_input[i] = (unsigned int)i * 2654435761u + 12345u;",
    "zeros":       "h_input[i] = 0u;",
    "small":       "h_input[i] = (unsigned int)(i % 17);",
    "random":      "h_input[i] = (unsigned int)((i * 1103515245u + 12345u) ^ (i >> 3));",
    "adversarial": "h_input[i] = ((i & 1) ? 0xFFFFFFFFu : 0u) ^ ((i & 2) ? 0x80000000u : 0u);",
}


def _scalar_arg_literal(t: GType, idx: int) -> str:
    """A deterministic scalar value for the idx-th extra kernel parameter."""
    seed = idx * 7 + 42
    if isinstance(t, TSChar):
        return f"((signed char){((seed * 37) % 255) - 128})"
    if isinstance(t, TChar):
        return f"((unsigned char){(seed * 37) % 256}u)"
    if isinstance(t, TShort):
        return f"((short){((seed * 2654435761) % 65535) - 32768})"
    if isinstance(t, TUShort):
        return f"((unsigned short){(seed * 2654435761) % 65536}u)"
    if isinstance(t, TInt):
        return f"{(seed * 2654435761) % 2147483647}"
    if isinstance(t, TUInt):
        return f"{(seed * 2654435761) % 4294967295}u"
    if isinstance(t, TLong):
        return f"{(seed * 6364136223846793005) % 9223372036854775807}ll"
    if isinstance(t, TULong):
        return f"{(seed * 6364136223846793005) % 18446744073709551615}ull"
    if isinstance(t, TFloat):
        return f"{(((seed * 1103515245 + 12345) % 2000) / 64.0 - 15.0):.6f}f"
    if isinstance(t, TDouble):
        return f"{(((seed * 1103515245 + 12345) % 4000) / 32.0 - 60.0):.6f}"
    if isinstance(t, THalf):
        v = ((seed * 1103515245 + 12345) % 2000) / 64.0 - 15.0
        return f"__float2half_rn({v:.6f}f)"
    if isinstance(t, TBF16):
        v = ((seed * 1103515245 + 12345) % 4000) / 32.0 - 60.0
        return f"__float2bfloat16({v:.6f}f)"
    if isinstance(t, TBool):
        return "true" if (seed % 2) else "false"
    return str(seed)


def _scalar_arg_bytes(t: GType, idx: int) -> bytes:
    """Encode the fixed host-harness scalar as CUDA launch-parameter bytes.

    Keeping this next to ``_scalar_arg_literal`` makes the Runtime and Driver
    harnesses share one deterministic argument contract.  All generated
    scalar values are finite, so Python's IEEE conversions match the CUDA
    round-to-nearest intrinsics used by the emitted host code.
    """
    seed = idx * 7 + 42
    if isinstance(t, TSChar):
        return struct.pack("<b", ((seed * 37) % 255) - 128)
    if isinstance(t, TChar):
        return struct.pack("<B", (seed * 37) % 256)
    if isinstance(t, TShort):
        return struct.pack("<h", ((seed * 2654435761) % 65535) - 32768)
    if isinstance(t, TUShort):
        return struct.pack("<H", (seed * 2654435761) % 65536)
    if isinstance(t, TInt):
        return struct.pack("<i", (seed * 2654435761) % 2147483647)
    if isinstance(t, TUInt):
        return struct.pack("<I", (seed * 2654435761) % 4294967295)
    if isinstance(t, TLong):
        return struct.pack("<q", (seed * 6364136223846793005) % 9223372036854775807)
    if isinstance(t, TULong):
        return struct.pack("<Q", (seed * 6364136223846793005) % 18446744073709551615)
    if isinstance(t, TFloat):
        value = float(f"{(((seed * 1103515245 + 12345) % 2000) / 64.0 - 15.0):.6f}")
        return struct.pack("<f", value)
    if isinstance(t, TDouble):
        value = float(f"{(((seed * 1103515245 + 12345) % 4000) / 32.0 - 60.0):.6f}")
        return struct.pack("<d", value)
    if isinstance(t, THalf):
        value = float(f"{(((seed * 1103515245 + 12345) % 2000) / 64.0 - 15.0):.6f}")
        value = struct.unpack("<f", struct.pack("<f", value))[0]
        return struct.pack("<e", value)
    if isinstance(t, TBF16):
        value = float(f"{(((seed * 1103515245 + 12345) % 4000) / 32.0 - 60.0):.6f}")
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        # CUDA __float2bfloat16 uses round-to-nearest-even.
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        return struct.pack("<H", rounded >> 16)
    if isinstance(t, TBool):
        return struct.pack("<B", 1 if (seed % 2) else 0)
    raise TypeError(f"unsupported generator_ir scalar parameter type {t!r}")


def _dim3(dims, name: str) -> str:
    x, y, z = dims
    if y == 1 and z == 1:
        return f"dim3 {name}({x});"
    if z == 1:
        return f"dim3 {name}({x}, {y});"
    return f"dim3 {name}({x}, {y}, {z});"


class HostHarness:
    """Emits ``int main(void)`` for a kernel under a launch configuration."""

    def __init__(self, w: SourceWriter, api: RuntimeApi,
                 cfg: LaunchConfig, type_name=None) -> None:
        self.w = w
        self.api = api
        self.cfg = cfg
        self._type_name = type_name or (lambda t: t.spelling())

    # -- argument mapping ----------------------------------------------------

    def _parameter_bindings(self, kernel: Func) -> List[Tuple[str, Any, int]]:
        """Return (role, parameter, scalar-index) in kernel ABI order."""
        params = kernel.params
        if not params:
            raise ValueError("generator_ir Driver mode requires explicit kernel parameters")
        bindings: List[Tuple[str, Any, int]] = []
        have_out = have_in = have_n = False
        scalar_idx = 0
        for p in params:
            pt, pn = p.gtype, p.name.lower()
            if isinstance(pt, TPointer):
                if p.name == "output_fp":
                    role = "output_fp"
                elif not have_out and ("out" in pn or "result" in pn or "dst" in pn):
                    role = "output"
                    have_out = True
                elif not have_in and ("in" in pn or "src" in pn or "data" in pn):
                    role = "input"
                    have_in = True
                elif not have_out:
                    role = "output"
                    have_out = True
                elif not have_in:
                    role = "input"
                    have_in = True
                else:
                    role = "input"
                bindings.append((role, p, -1))
            elif not have_n:
                bindings.append(("n", p, -1))
                have_n = True
            else:
                bindings.append(("scalar", p, scalar_idx))
                scalar_idx += 1
        return bindings

    def kernel_args(self, kernel: Func) -> str:
        """Map kernel params to host variables; extra scalars get fixed values."""
        args: List[str] = []
        for role, p, scalar_idx in self._parameter_bindings(kernel):
            if role == "output":
                args.append("d_output")
            elif role == "input":
                args.append("d_input")
            elif role == "output_fp":
                args.append("d_output_fp")
            elif role == "n":
                args.append("n")
            else:
                args.append(_scalar_arg_literal(p.gtype, scalar_idx))
        return ", ".join(args)

    def driver_spec(self, kernel: Func, program) -> Dict[str, Any]:
        """Materialize the variable CUDA kernel ABI for the native runner."""
        from generator_ir.ir.statements import Declare
        from generator_ir.ir.typedecl import TArray, MemSpace, Qual

        n = int(self.cfg.n_elements) if self.cfg.n_elements > 0 else max(
            self.cfg.threads_per_block * self.cfg.block_count, 1024)
        buf_size = n * n if self.cfg.kx_is_2d else n
        in_size = buf_size * max(1, int(self.cfg.input_size_mult))
        args: List[Dict[str, Any]] = []
        for role, p, scalar_idx in self._parameter_bindings(kernel):
            if role == "scalar":
                raw = _scalar_arg_bytes(p.gtype, scalar_idx)
                args.append({
                    "kind": "scalar", "name": p.name,
                    "type": p.gtype.spelling(), "data_hex": raw.hex(),
                })
            elif role == "output_fp":
                if not isinstance(p.gtype, TPointer) or p.gtype.base.width not in (4, 8):
                    raise ValueError(f"unsupported output_fp parameter {p.gtype!r}")
                args.append({
                    "kind": role, "name": p.name,
                    "element_bytes": p.gtype.base.width,
                })
            else:
                args.append({"kind": role, "name": p.name})

        globals_spec: List[Dict[str, Any]] = []
        for d in program.decls:
            if not (isinstance(d, Declare) and isinstance(d.gtype, TArray)):
                continue
            if d.space in (MemSpace.GLOBAL, MemSpace.GENERIC) or Qual.DEVICE in d.quals:
                globals_spec.append({"name": d.name, "init": "zero"})
        if self.cfg.needs_constant_mem:
            globals_spec.append({"name": "_cmem", "init": "cmem_u32"})

        return {
            "version": 1,
            "kind": "generator_ir_checksum",
            "kernel": kernel.name,
            "args": args,
            "globals": globals_spec,
            "grid": list(self.cfg.grid),
            "block": list(self.cfg.block),
            "shared_bytes": int(self.cfg.shared_bytes),
            "n": n,
            "buf_size": buf_size,
            "in_size": in_size,
            "input_distribution": self.cfg.input_dist if self.cfg.input_dist in _INPUT_INIT else "hash",
            "output_element_bytes": 4,
            "checksum": "boost_u32",
        }
    def _fp_param(self, kernel: Func):
        for p in kernel.params:
            if p.name == "output_fp" and isinstance(p.gtype, TPointer):
                if isinstance(p.gtype.base, (TFloat, TDouble)):
                    return p
        return None

    def _n_expr(self) -> str:
        cfg = self.cfg
        if cfg.n_elements > 0:
            return str(cfg.n_elements)
        total = cfg.threads_per_block * cfg.block_count
        return str(max(total, 1024))

    def _device_array_zero_init(self, program) -> None:
        """Zero __device__ globals via MemcpyToSymbol (determinism guard)."""
        from generator_ir.ir.statements import Declare
        from generator_ir.ir.typedecl import TArray, MemSpace, Qual
        w, api, check = self.w, self.api, self.api.error_check_call
        wrote_header = False
        for d in program.decls:
            if not (isinstance(d, Declare) and isinstance(d.gtype, TArray)):
                continue
            if not (d.space in (MemSpace.GLOBAL, MemSpace.GENERIC)
                    or Qual.DEVICE in d.quals):
                continue
            total = 1
            inner = d.gtype
            while isinstance(inner, TArray):
                total *= inner.count
                inner = inner.elem
            bytes_expr = f"{total} * sizeof({self._type_name(inner)})"
            if not wrote_header:
                w.line("// Zero-initialize __device__ arrays for determinism")
                wrote_header = True
            w.line("{")
            w.kx_indent()
            w.line(f"static char _zeros_{d.name}[{bytes_expr}];")
            w.line(f"memset(_zeros_{d.name}, 0, {bytes_expr});")
            w.line(f"{check}({api.memcpy_to_symbol}({d.name}, _zeros_{d.name}, {bytes_expr}));")
            w.kx_dedent()
            w.line("}")
        if wrote_header:
            w.blank()

    # -- main entry ------------------------------------------------------------

    def emit(self, kernel: Func, program=None) -> None:
        w, api, cfg = self.w, self.api, self.cfg
        check = api.error_check_call

        kernel_name = kernel.name
        shared = cfg.shared_bytes
        launch_args = self.kernel_args(kernel)
        shared_arg = "shared_bytes" if shared > 0 else "0"
        launch = f"{kernel_name}<<<grid, block, {shared_arg}>>>({launch_args})"

        w.line("int main(void) {")
        w.kx_indent()

        # Sizes
        w.line(f"const int n = {self._n_expr()};")
        w.line(f"const int buf_size = {'n * n' if cfg.kx_is_2d else 'n'};")
        mult = max(1, int(cfg.input_size_mult))
        if mult > 1:
            w.line(f"const int in_size = buf_size * {mult};")
        else:
            w.line("const int in_size = buf_size;")
        w.blank()

        fp_param = self._fp_param(kernel)

        # Host allocations
        w.line("// Host allocations")
        w.line("unsigned int *h_input = (unsigned int*)malloc(in_size * sizeof(unsigned int));")
        w.line("unsigned int *h_output = (unsigned int*)malloc(buf_size * sizeof(unsigned int));")
        w.line('if (!h_input || !h_output) { fprintf(stderr, "malloc failed\\n"); return 1; }')
        if fp_param is not None:
            fty = "double" if isinstance(fp_param.gtype.base, TDouble) else "float"
            w.line(f"{fty} *h_output_fp = ({fty}*)malloc(buf_size * sizeof({fty}));")
            w.line('if (!h_output_fp) { fprintf(stderr, "malloc failed\\n"); return 1; }')
        w.blank()

        # Deterministic input fill (deferred into the run_pass lambda emitted below,
        # so the input-sensitivity re-run can reuse this process instead of forking
        # a second one and paying CUDA context init again).
        dist = cfg.input_dist if cfg.input_dist in _INPUT_INIT else "hash"
        w.line(f"// Deterministic input initialization (dist={dist}) happens in run_pass()")
        w.line('const char *input_salt_text = getenv("CUDA_STRESS_INPUT_SALT");')
        w.line("const unsigned long long input_salt = "
               "(input_salt_text && *input_salt_text) "
               "? strtoull(input_salt_text, NULL, 0) : 0ULL;")
        w.blank()

        # Device allocations
        w.line("// Device allocations")
        w.line("unsigned int *d_input = NULL, *d_output = NULL;")
        w.line(f"{check}({api.malloc}((void**)&d_input, in_size * sizeof(unsigned int)));")
        w.line(f"{check}({api.malloc}((void**)&d_output, buf_size * sizeof(unsigned int)));")
        if fp_param is not None:
            fty = "double" if isinstance(fp_param.gtype.base, TDouble) else "float"
            w.line(f"{fty} *d_output_fp = NULL;")
            w.line(f"{check}({api.malloc}((void**)&d_output_fp, buf_size * sizeof({fty})));")
            w.line(f"{check}({api.memset}(d_output_fp, 0, buf_size * sizeof({fty})));")
        w.blank()

        # Constant pool
        if cfg.needs_constant_mem:
            w.line("// __constant__ pool init (deterministic)")
            w.line("unsigned int h_cmem[256];")
            w.line("for (int i = 0; i < 256; i++) { h_cmem[i] = (unsigned int)i * 2654435761u + 40503u; }")
            w.line(f"{check}({api.memcpy_to_symbol}(_cmem, h_cmem, 256 * sizeof(unsigned int)));")
            w.blank()

        # Zero-init __device__ globals (before any launch)
        if program is not None:
            self._device_array_zero_init(program)

        # Launch config
        w.line("// Launch configuration")
        w.line(_dim3(cfg.grid, "grid"))
        w.line(_dim3(cfg.block, "block"))
        if shared > 0:
            w.line(f"const int shared_bytes = {shared};")
        w.blank()

        # Timing harness
        if cfg.checksum_only:
            w.line("// Checksum-only mode: skip all warmup and timing launches")
        elif cfg.time_profiler:
            self._harness_profiler(launch)
        elif cfg.time_clock64:
            self._harness_clock64(launch)
        elif cfg.time_loop:
            self._harness_loop(launch)
        elif cfg.time_ci:
            self._harness_ci(launch)
        else:
            self._harness_simple(launch)
        w.blank()

        # Deterministic checksum pass, wrapped in a lambda so both the normal
        # (salt=0) run and the input-sensitivity (salt=1) run share one process
        # and one CUDA context.  Each invocation prints one "checksum=<u64>" line.
        w.line("// Deterministic checksum pass (run_pass is reusable in-process)")
        w.line("auto run_pass = [&](unsigned long long salt) -> int {")
        w.kx_indent()
        w.line("for (int i = 0; i < in_size; i++) {")
        w.kx_indent()
        w.line(_INPUT_INIT[dist])
        w.line("if (salt != 0ULL) {")
        w.kx_indent()
        w.line("unsigned long long z = salt + 0x9e3779b97f4a7c15ULL "
               "* ((unsigned long long)(unsigned int)i + 1ULL);")
        w.line("z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;")
        w.line("z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;")
        w.line("z ^= z >> 31;")
        w.line("unsigned int salt_word = (unsigned int)(z ^ (z >> 32));")
        w.line("h_input[i] ^= salt_word ? salt_word : 0xa5a5a5a5u;")
        w.kx_dedent()
        w.line("}")
        w.kx_dedent()
        w.line("}")
        w.line(f"{check}({api.memcpy}(d_input, h_input, in_size * sizeof(unsigned int), {api.memcpy_h2d}));")
        w.line(f"{check}({api.memset}(d_output, 0, buf_size * sizeof(unsigned int)));")
        w.line(f"{launch};")
        w.line(f"{check}({api.get_last_error}());")
        w.line(f"{check}({api.device_sync}());")
        w.line(f"{check}({api.memcpy}(h_output, d_output, buf_size * sizeof(unsigned int), {api.memcpy_d2h}));")
        w.line("unsigned long long hash = 42;")
        w.line("for (int i = 0; i < buf_size; i++) {")
        w.kx_indent()
        w.line("unsigned long long val = h_output[i];")
        w.line("hash ^= val + 0x9e3779b9ULL + (hash << 6) + (hash >> 2);")
        w.kx_dedent()
        w.line("}")
        w.line('printf("checksum=%llu\\n", hash);')
        w.line("return 0;")
        w.kx_dedent()
        w.line("};")
        w.blank()
        w.line("run_pass(input_salt);")
        w.line('if (getenv("CUDA_STRESS_ALSO_SALT")) { run_pass(1ULL); }')
        w.blank()

        # Cleanup
        w.line("// Cleanup")
        if cfg.checksum_only or cfg.time_profiler or cfg.time_clock64:
            pass
        # time_loop outranks time_ci in the harness dispatch above, so only the
        # CI harness (cal_start/cal_stop) ran when CI is set AND loop is not.
        elif cfg.time_ci and not cfg.time_loop:
            w.line(f"{check}({api.event_destroy}(cal_start));")
            w.line(f"{check}({api.event_destroy}(cal_stop));")
        else:
            w.line(f"{check}({api.event_destroy}(start));")
            w.line(f"{check}({api.event_destroy}(stop));")
        w.line(f"{check}({api.free}(d_input));")
        w.line(f"{check}({api.free}(d_output));")
        if fp_param is not None:
            w.line(f"{check}({api.free}(d_output_fp));")
        w.line("free(h_input);")
        w.line("free(h_output);")
        if fp_param is not None:
            w.line("free(h_output_fp);")
        w.line("return 0;")
        w.kx_dedent()
        w.line("}")

    # -- timing harnesses -------------------------------------------------------

    def _harness_simple(self, launch: str) -> None:
        """Delegate to timing_modes.harness_simple_mode."""
        from generator_ir.emit.timing_modes import harness_simple_mode
        harness_simple_mode(self.w, self.api, launch)

    def _harness_profiler(self, launch: str) -> None:
        """Delegate to timing_modes.harness_profiler_mode."""
        from generator_ir.emit.timing_modes import harness_profiler_mode
        harness_profiler_mode(self.w, self.api, launch)

    def _harness_ci(self, launch: str) -> None:
        """Delegate to timing_modes.harness_ci_mode."""
        from generator_ir.emit.timing_modes import harness_ci_mode
        harness_ci_mode(self.w, self.api, launch)

    def _harness_loop(self, launch: str) -> None:
        """Delegate to timing_modes.harness_loop_mode."""
        from generator_ir.emit.timing_modes import harness_loop_mode
        harness_loop_mode(self.w, self.api, launch)

    def _harness_clock64(self, launch: str) -> None:
        """Delegate to timing_modes.harness_clock64_mode."""
        from generator_ir.emit.timing_modes import harness_clock64_mode
        harness_clock64_mode(self.w, self.api, launch)


def cuda_driver_spec(program, kernel: Func) -> Dict[str, Any]:
    """Public adapter used by the pipeline's generator_ir strategy."""
    return HostHarness(SourceWriter(), CUDA_API, program.launch).driver_spec(kernel, program)
