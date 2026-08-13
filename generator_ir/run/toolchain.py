"""Compiler wrappers and output parsing for CUDA and HIP."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class CompileResult:
    success: bool
    output_path: Optional[str]
    stderr: str
    compile_time_s: float
    command: str = ""
    source_path: str = ""


@dataclass
class RunResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    checksum: Optional[int] = None
    avg_time_ms: Optional[float] = None
    min_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None
    all_times_ms: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    # CI harness fields
    ci_mean_us: Optional[float] = None
    ci_margin_us: Optional[float] = None
    ci_batch_size: Optional[int] = None
    # clock64 fields
    cycles_median: Optional[int] = None
    cycles_all: List[int] = field(default_factory=list)
    # loop-mode field
    loop_n: Optional[int] = None


# ---------------------------------------------------------------------------
# stdout parsing
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"^(\w+)=(.+)$")
_CI_RE = re.compile(r"^([A-Z_]+):\s*(.+)$")


def parse_kv(stdout: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def extract_ci_fields(stdout: str) -> Dict[str, str]:
    """KERNEL_TIME_*: value lines (accept one-or-more spaces after colon)."""
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        m = _CI_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def collect_time_lines(stdout: str) -> List[float]:
    out: List[float] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("time_ms="):
            try:
                out.append(float(line.split("=", 1)[1]))
            except ValueError:
                pass
    return out


def collect_cycle_lines(stdout: str) -> List[int]:
    out: List[int] = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "CYCLES":
            try:
                out.append(int(parts[1]))
            except ValueError:
                pass
    return out


def _kx_to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _kx_to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def decode_binary_output(stdout: str) -> dict:
    """One-stop parse of a harness binary's stdout."""
    kv = parse_kv(stdout)
    ci = extract_ci_fields(stdout)
    times = collect_time_lines(stdout)
    cycles = collect_cycle_lines(stdout)

    checksum = _kx_to_int(kv["checksum"]) if "checksum" in kv else None
    avg = _kx_to_float(kv["avg_time_ms"]) if "avg_time_ms" in kv else None
    if avg is None and times:
        avg = sum(times) / len(times)

    cycles_median = None
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "CLOCK64_MEDIAN_CYCLES":
            cycles_median = _kx_to_int(parts[1])
            break
    if cycles_median is None and len(cycles) >= 10:
        s = sorted(cycles)
        cycles_median = s[len(s) // 2]

    loop_n = None
    for line in stdout.splitlines():
        if line.strip().startswith("LOOP_N:"):
            loop_n = _kx_to_int(line.split(":", 1)[1].strip())
            break

    return {
        "checksum": checksum,
        "avg_time_ms": avg,
        "min_time_ms": (_kx_to_float(kv["min_time_ms"]) if "min_time_ms" in kv
                        else (min(times) if times else None)),
        "max_time_ms": (_kx_to_float(kv["max_time_ms"]) if "max_time_ms" in kv
                        else (max(times) if times else None)),
        "all_times_ms": times,
        "ci_mean_us": _kx_to_float(ci.get("KERNEL_TIME_MEAN_US", "")),
        "ci_margin_us": _kx_to_float(ci.get("KERNEL_TIME_MARGIN_US", "")),
        "ci_batch_size": _kx_to_int(ci.get("KERNEL_TIME_BATCH_SIZE", "")),
        "cycles_median": cycles_median,
        "cycles_all": cycles,
        "loop_n": loop_n,
    }


# ---------------------------------------------------------------------------
# Shared harness helpers (used by both cross-compiler pipelines)
# ---------------------------------------------------------------------------

def pull_cycles(rr: RunResult) -> Optional[int]:
    if rr.cycles_median is not None:
        return rr.cycles_median
    if len(rr.cycles_all) >= 10:
        s = sorted(rr.cycles_all)
        return s[len(s) // 2]
    return None


def _kx_same_source_checksum_error(old: RunResult, new: RunResult) -> Optional[str]:
    for label, rr in (("old", old), ("new", new)):
        if not rr.success:
            return f"{label} compiler execution failed (rc={rr.return_code})"
    if old.checksum is None:
        return "old compiler execution produced no checksum"
    if new.checksum is None:
        return "new compiler execution produced no checksum"
    if old.checksum == new.checksum:
        return None
    return (f"checksum mismatch for identical source: "
            f"old={old.checksum} new={new.checksum}")


def _purge_binaries(*crs: Optional[CompileResult]) -> None:
    """Delete compiled executables (sources stay for reproducibility)."""
    for cr in crs:
        path = getattr(cr, "output_path", None)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


def _write_assemble_args_once(work_dir: Path, cr_old: CompileResult,
                             cr_new: CompileResult, opt_level: str) -> None:
    """Record old/new assemble commands once per campaign for reproducibility."""
    path = work_dir / "compile_args.txt"
    if path.exists():
        return
    if not getattr(cr_old, "command", None) or not getattr(cr_new, "command", None):
        return
    text = (
        "# Compile commands used in this campaign.\n"
        "# Flags are identical for every test case; only the per-case source\n"
        "# path and -o output path differ.\n"
        f"# opt level: {opt_level}\n\n"
        "[old compiler]\n" + cr_old.command + "\n\n"
        "[new compiler]\n" + cr_new.command + "\n"
    )
    try:
        path.write_text(text)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Compiler base
# ---------------------------------------------------------------------------

class BaseCompiler:
    """Write source, assemble with an arch flag, run the binary, parse output."""

    def __init__(self, compiler_path: str, arch: str, extension: str,
                 arch_flag_prefix: str,
                 extra_flags: Optional[List[str]] = None) -> None:
        self.compiler_path = compiler_path
        self.isa = arch
        self.extension = extension
        self.isa_flag_prefix = arch_flag_prefix
        self.extra_flags = list(extra_flags or [])

    @property
    def kx_arch_flag(self) -> str:
        return f"{self.isa_flag_prefix}{self.isa}"

    def _workdir(self, output_dir: Optional[str]) -> str:
        if output_dir is None:
            return tempfile.mkdtemp(prefix="gpuperf_")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def assemble(self, source: str, opt_level: str = "-O3",
                output_dir: Optional[str] = None,
                extra_flags: Optional[List[str]] = None,
                basename: str = "kernel",
                timeout: int = 120) -> CompileResult:
        out_dir = self._workdir(output_dir)
        src_path = os.path.join(out_dir, f"{basename}{self.extension}")
        exe_path = os.path.join(out_dir, basename)
        try:
            with open(src_path, "w") as fh:
                fh.write(source)
        except OSError as exc:
            return CompileResult(False, None, f"write failed: {exc}", 0.0,
                                 source_path=src_path)

        cmd = [self.compiler_path, src_path, "-o", exe_path, opt_level,
               self.kx_arch_flag, "-std=c++17"]
        cmd.extend(self.extra_flags)
        if extra_flags:
            cmd.extend(extra_flags)
        cmd_str = " ".join(cmd)
        log.debug("assemble: %s", cmd_str)

        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return CompileResult(False, None,
                                 f"assemble timed out ({timeout}s)", float(timeout),
                                 command=cmd_str, source_path=src_path)
        except FileNotFoundError:
            return CompileResult(False, None,
                                 f"compiler not found: {self.compiler_path}",
                                 0.0, command=cmd_str, source_path=src_path)
        except OSError as exc:
            # Mirrors run(): PermissionError / fork failures etc. surface as a
            # failed CompileResult rather than propagating past _purge_binaries.
            return CompileResult(False, None, f"assemble failed: {exc}",
                                 0.0, command=cmd_str, source_path=src_path)
        dt = time.monotonic() - t0
        ok = proc.returncode == 0
        if not ok:
            log.debug("assemble failed: %s", proc.stderr[:400])
        return CompileResult(ok, exe_path if ok else None, proc.stderr, dt,
                             command=cmd_str, source_path=src_path)

    def run(self, exe_path: str, timeout: int = 300,
            env_extra: Optional[Dict[str, str]] = None) -> RunResult:
        if not os.path.isfile(exe_path):
            return RunResult(False, "", f"executable not found: {exe_path}", -1)
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        t0 = time.monotonic()
        try:
            proc = subprocess.run([exe_path], capture_output=True, text=True,
                                  timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return RunResult(False, "", f"run timed out ({timeout}s)", -1,
                             wall_time_s=float(timeout))
        except OSError as exc:
            return RunResult(False, "", str(exc), -1)
        wall = time.monotonic() - t0

        parsed = decode_binary_output(proc.stdout)
        ok = proc.returncode == 0
        if not ok:
            log.debug("run failed rc=%d: %s", proc.returncode,
                      proc.stderr[:300])
        return RunResult(
            ok, proc.stdout, proc.stderr, proc.returncode,
            checksum=parsed["checksum"],
            avg_time_ms=parsed["avg_time_ms"],
            min_time_ms=parsed["min_time_ms"],
            max_time_ms=parsed["max_time_ms"],
            all_times_ms=parsed["all_times_ms"],
            wall_time_s=wall,
            ci_mean_us=parsed["ci_mean_us"],
            ci_margin_us=parsed["ci_margin_us"],
            ci_batch_size=parsed["ci_batch_size"],
            cycles_median=parsed["cycles_median"],
            cycles_all=parsed["cycles_all"],
            loop_n=parsed["loop_n"],
        )

    def assemble_and_execute(self, source: str, opt_level: str = "-O3",
                        output_dir: Optional[str] = None,
                        run_timeout: int = 300):
        cr = self.assemble(source, opt_level, output_dir)
        if not cr.success:
            return cr, None
        return cr, self.run(cr.output_path, timeout=run_timeout)

    @staticmethod
    def cleanup(output_dir: str) -> None:
        shutil.rmtree(output_dir, ignore_errors=True)


class CudaCompiler(BaseCompiler):
    def __init__(self, nvcc_path: str = "/usr/local/cuda/bin/nvcc",
                 arch: str = "sm_75",
                 extra_flags: Optional[List[str]] = None) -> None:
        super().__init__(nvcc_path, arch, ".cu", "-arch=", extra_flags)

    def run(self, exe_path: str, timeout: int = 300,
            env_extra: Optional[Dict[str, str]] = None) -> RunResult:
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        if env_extra:
            env.update(env_extra)
        return super().run(exe_path, timeout=timeout, env_extra=env)


class HipCompiler(BaseCompiler):
    def __init__(self, hipcc_path: str = "/opt/rocm/bin/hipcc",
                 arch: str = "gfx908",
                 extra_flags: Optional[List[str]] = None) -> None:
        super().__init__(hipcc_path, arch, ".hip", "--offload-arch=",
                         extra_flags)

    def run(self, exe_path: str, timeout: int = 300,
            env_extra: Optional[Dict[str, str]] = None) -> RunResult:
        env = {"HIP_VISIBLE_DEVICES": "0", "AMD_LOG_LEVEL": "0"}
        if env_extra:
            env.update(env_extra)
        return super().run(exe_path, timeout=timeout, env_extra=env)


def create_compiler(backend: str = "hip", arch: Optional[str] = None,
                  compiler_path: Optional[str] = None,
                  extra_flags: Optional[List[str]] = None) -> BaseCompiler:
    backend = backend.lower()
    if backend == "hip":
        return HipCompiler(hipcc_path=compiler_path or "/opt/rocm/bin/hipcc",
                           arch=arch or "gfx908", extra_flags=extra_flags)
    if backend == "cuda":
        return CudaCompiler(nvcc_path=compiler_path or "/usr/local/cuda/bin/nvcc",
                            arch=arch or "sm_75", extra_flags=extra_flags)
    raise ValueError(f"unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Parallel assemble / run
# ---------------------------------------------------------------------------

def _compile_job(args: tuple) -> Tuple[str, CompileResult]:
    (label, compiler_path, arch, source, opt_level, extra_flags,
     ext, arch_prefix) = args
    compiler = BaseCompiler(compiler_path, arch, ext, arch_prefix)
    return label, compiler.assemble(source, opt_level, extra_flags=extra_flags)


def parallel_assemble(jobs: Dict[str, dict], max_workers: int = 6) -> Dict[str, CompileResult]:
    """Compile many variants concurrently (compilation is CPU-bound)."""
    items = [
        (label, spec["compiler_path"], spec["arch"], spec["source_code"],
         spec["opt_level"], spec.get("extra_flags", []), spec["ext"],
         spec["arch_prefix"])
        for label, spec in jobs.items()
    ]
    results: Dict[str, CompileResult] = {}
    if not items:
        return results
    if len(items) == 1:
        label, cr = _compile_job(items[0])
        results[label] = cr
        return results
    with ProcessPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = {pool.submit(_compile_job, item): item[0] for item in items}
        for fut in futures:
            label = futures[fut]
            try:
                lbl, cr = fut.result()
                results[lbl] = cr
            except Exception as exc:
                log.error("parallel assemble failed for %s: %s", label, exc)
                results[label] = CompileResult(
                    False, None, f"parallel assemble exception: {exc}", 0.0)
    return results


def parallel_execute(jobs: Dict[str, dict], max_workers: int = 1) -> Dict[str, RunResult]:
    """Run binaries.  GPUs serialize kernels, so this is sequential by default."""
    results: Dict[str, RunResult] = {}
    for label, spec in jobs.items():
        compiler = BaseCompiler(spec["compiler_path"], spec["arch"],
                                spec["ext"], spec["arch_prefix"])
        try:
            results[label] = compiler.run(
                spec["exe_path"], timeout=spec.get("timeout", 300),
                env_extra=spec.get("env_extra"))
        except Exception as exc:
            log.error("run failed for %s: %s", label, exc)
            results[label] = RunResult(False, "", f"run exception: {exc}", -1)
    return results


# Backward-compatible aliases.
parse_run_output = decode_binary_output
parse_ci = extract_ci_fields
parse_time_samples = collect_time_lines
parse_cycles = collect_cycle_lines
