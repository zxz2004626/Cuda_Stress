"""Compile a generated testcase into the N x M matrix of PTX / CUBIN / SASS artifacts.

Pipeline per (toolchain, ptxas-profile):
    nvcc -arch=sm_86 -ccbin <cc> -ptx test.cu -> kernel.ptx   (once per toolchain)
    ptxas -arch=sm_86 -O<n> -v kernel.ptx   -> kernel.cubin  (per profile)
    cuobjdump -sass kernel.cubin            -> kernel.sass   (per profile)

Every invocation records argv, stdout, stderr, return code and elapsed seconds in
command.json, so "did the -O flag really reach ptxas" is auditable.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, OptimizationProfile, Toolchain, isolated_env
from . import disasm

DEFAULT_TIMEOUT = 180


@dataclass
class CmdRecord:
    argv: List[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0
    ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": self.argv, "returncode": self.returncode,
            "stdout": self.stdout[-8000:], "stderr": self.stderr[-8000:],
            "elapsed_s": round(self.elapsed, 3), "ok": self.ok,
        }


def _exec(argv: List[str], env: Dict[str, str], timeout: int) -> CmdRecord:
    t0 = time.monotonic()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        rec = CmdRecord(argv=list(argv), returncode=r.returncode,
                        stdout=r.stdout, stderr=r.stderr,
                        elapsed=time.monotonic() - t0, ok=r.returncode == 0)
    except subprocess.TimeoutExpired:
        rec = CmdRecord(argv=list(argv), returncode=124, elapsed=timeout,
                        stderr=f"[compile timeout after {timeout}s]", ok=False)
    except Exception as e:  # noqa: BLE001
        rec = CmdRecord(argv=list(argv), returncode=-1, stderr=str(e), ok=False)
    return rec


def nvcc_ptx(tc: Toolchain, src_path: Path, out_ptx: Path,
             timeout: Optional[int] = None) -> CmdRecord:
    env = isolated_env(tc)
    argv = [tc.nvcc, f"-arch={tc.arch}", "-ccbin", tc.ccbin,
            "-ptx", "-o", str(out_ptx), str(src_path)]
    return _exec(argv, env, timeout or DEFAULT_TIMEOUT)


def nvcc_standalone_ptx(tc: Toolchain, src_path: Path, out_ptx: Path,
                        timeout: Optional[int] = None) -> CmdRecord:
    """Best-effort PTX extraction for a complete CUDA Runtime program."""
    env = isolated_env(tc)
    argv = [tc.nvcc, f"-arch={tc.arch}", "-ccbin", tc.ccbin,
            "-std=c++17", "-ptx", "-o", str(out_ptx), str(src_path)]
    return _exec(argv, env, timeout or DEFAULT_TIMEOUT)


def nvcc_standalone_executable(tc: Toolchain, profile: OptimizationProfile,
                               src_path: Path, out_exe: Path,
                               extra_flags: Optional[List[str]] = None,
                               timeout: Optional[int] = None) -> CmdRecord:
    """Compile a self-contained CUDA testcase while pinning ptxas -O<n>.

    Host optimization remains fixed at -O2 for every profile; only the ptxas
    optimization level varies, matching the legacy pipeline's comparison axis.
    """
    env = isolated_env(tc)
    argv = [tc.nvcc, f"-arch={tc.arch}", "-ccbin", tc.ccbin,
            "-std=c++17", "-O2",
            "-Xptxas", f"-O{profile.ptxas_opt}",
            "-Xptxas", "-v"]
    for flag in extra_flags or []:
        argv.extend(["-Xptxas", str(flag)])
    argv += ["-o", str(out_exe), str(src_path)]
    return _exec(argv, env, timeout or DEFAULT_TIMEOUT)


def ptxas_cubin(tc: Toolchain, profile: OptimizationProfile, ptx_path: Path,
                out_cubin: Path, extra_flags: Optional[List[str]] = None,
                timeout: Optional[int] = None) -> CmdRecord:
    env = isolated_env(tc)
    argv = [tc.ptxas, f"-arch={tc.arch}", f"-O{profile.ptxas_opt}", "-v"]
    if extra_flags:
        argv.extend(extra_flags)
    argv += ["-o", str(out_cubin), str(ptx_path)]
    return _exec(argv, env, timeout or DEFAULT_TIMEOUT)


def cuobjdump_sass(tc: Toolchain, cubin_path: Path, out_sass: Path,
                   timeout: Optional[int] = None) -> CmdRecord:
    env = isolated_env(tc)
    argv = [tc.cuobjdump, "-sass", str(cubin_path)]
    rec = _exec(argv, env, timeout or DEFAULT_TIMEOUT)
    if rec.ok:
        out_sass.write_text(rec.stdout)
    return rec


def write_command_json(build_dir: Path, rec: CmdRecord) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "command.json").write_text(json.dumps(rec.to_dict(), indent=2))
    (build_dir / "stdout.txt").write_text(rec.stdout)
    (build_dir / "stderr.txt").write_text(rec.stderr)


def compile_build(config: Config, tc: Toolchain, profile: OptimizationProfile,
                  test_dir: Path, build_dir: Path,
                  extra_flags: Optional[List[str]] = None,
                  timeout: Optional[int] = None) -> Dict[str, Any]:
    """Compile one (toolchain x profile) build into build_dir. Returns compile_result dict."""
    src = test_dir / "test.cu"
    try:
        manifest = json.loads((test_dir / "manifest.json").read_text())
    except Exception:  # noqa: BLE001
        manifest = {}
    if manifest.get("execution_mode", "driver_kernel") == "standalone_checksum":
        return compile_standalone_build(
            config, tc, profile, test_dir, build_dir, extra_flags, timeout)

    ptx_path = build_dir / "kernel.ptx"
    cubin_path = build_dir / "kernel.cubin"
    sass_path = build_dir / "kernel.sass"
    build_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"toolchain": tc.id, "profile": profile.name,
                              "ok": False, "error": None, "commands": []}

    # 1. PTX (from this nvcc/cicc). Deterministic; compile fresh to guarantee the
    #    archived PTX is exactly what ptxas consumed.
    rec = nvcc_ptx(tc, src, ptx_path, timeout)
    write_command_json(build_dir, rec)
    result["commands"].append({"stage": "ptx", **rec.to_dict()})
    if not rec.ok:
        result["error"] = f"nvcc -ptx failed rc={rec.returncode}: {rec.stderr[-300:]}"
        return result

    # 2. CUBIN via ptxas with the requested -O level (proves the flag reaches ptxas).
    rec = ptxas_cubin(tc, profile, ptx_path, cubin_path, extra_flags, timeout)
    write_command_json(build_dir, rec)
    result["commands"].append({"stage": "cubin", **rec.to_dict()})
    if not rec.ok:
        result["error"] = f"ptxas -O{profile.ptxas_opt} failed rc={rec.returncode}: {rec.stderr[-300:]}"
        return result
    result["ptxas_verbose"] = rec.stderr
    result["registers"] = disasm.parse_ptxas_verbose(rec.stderr).get("registers")

    # 3. SASS disassembly.
    rec = cuobjdump_sass(tc, cubin_path, sass_path, timeout)
    result["commands"].append({"stage": "sass", **rec.to_dict()})
    if not rec.ok:
        result["error"] = f"cuobjdump -sass failed rc={rec.returncode}"
        return result

    result["ok"] = True
    return result


def compile_standalone_build(config: Config, tc: Toolchain,
                             profile: OptimizationProfile, test_dir: Path,
                             build_dir: Path,
                             extra_flags: Optional[List[str]] = None,
                             timeout: Optional[int] = None) -> Dict[str, Any]:
    """Compile + disassemble a generator-owned CUDA Runtime executable."""
    src = test_dir / "test.cu"
    ptx_path = build_dir / "kernel.ptx"
    exe_path = build_dir / "kernel.exe"
    sass_path = build_dir / "kernel.sass"
    build_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "toolchain": tc.id,
        "profile": profile.name,
        "execution_mode": "standalone_checksum",
        "ok": False,
        "error": None,
        "commands": [],
    }

    # PTX is an analysis artifact, not the executable input in this mode.  A
    # toolkit may decline -ptx for a full host program; retain that fact but do
    # not turn an otherwise runnable build into a compile failure.
    if ptx_path.exists():
        result["commands"].append({
            "stage": "ptx_analysis", "ok": True,
            "shared_artifact": True, "path": ptx_path.name,
        })
    else:
        ptx_rec = nvcc_standalone_ptx(
            tc, src, ptx_path,
            timeout=int(config.compilation.get("ptx_timeout", 60)))
        result["commands"].append({"stage": "ptx_analysis", **ptx_rec.to_dict()})
        if not ptx_rec.ok:
            ptx_path.unlink(missing_ok=True)

    rec = nvcc_standalone_executable(
        tc, profile, src, exe_path, extra_flags,
        timeout=timeout or int(config.compilation.get("compile_timeout", 120)))
    write_command_json(build_dir, rec)
    result["commands"].append({"stage": "executable", **rec.to_dict()})
    if not rec.ok:
        result["error"] = (
            f"nvcc standalone build failed rc={rec.returncode}: {rec.stderr[-300:]}")
        return result
    result["ptxas_verbose"] = rec.stderr
    result["registers"] = disasm.parse_ptxas_verbose(rec.stderr).get("registers")
    result["executable"] = exe_path.name

    rec = cuobjdump_sass(tc, exe_path, sass_path, timeout)
    result["commands"].append({"stage": "sass", **rec.to_dict()})
    if not rec.ok:
        result["error"] = f"cuobjdump -sass executable failed rc={rec.returncode}"
        return result

    result["ok"] = True
    return result


def compute_metrics(build_dir: Path, target_opcodes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read kernel.ptx / kernel.cubin / kernel.sass and produce stress metrics."""
    metrics: Dict[str, Any] = {}
    ptx_path = build_dir / "kernel.ptx"
    sass_path = build_dir / "kernel.sass"
    cubin_path = build_dir / "kernel.cubin"
    exe_path = build_dir / "kernel.exe"
    compile_json = build_dir / "compile_result.json"

    if ptx_path.exists():
        ptx_text = ptx_path.read_text()
        metrics["ptx_lines"] = len(ptx_text.splitlines())
        metrics["ptx_bytes"] = len(ptx_text)
        p = disasm.parse_ptx(ptx_text)
        metrics["ptx_instruction_count"] = p["instruction_count"]
        metrics["ptx_opcode_histogram"] = p["opcode_histogram"]
        metrics["ptx_normalized_hash"] = p["normalized_hash"]
        metrics["ptx_predicated_count"] = p["predicated_count"]
        # PTX target-opcode survival check
        if target_opcodes:
            tgt = [t for t in target_opcodes if t in ptx_text]
            metrics["ptx_target_opcodes_present"] = tgt

    if sass_path.exists():
        sass_text = sass_path.read_text()
        s = disasm.parse_sass(sass_text)
        metrics["sass_instruction_count"] = s["instruction_count"]
        metrics["sass_opcode_histogram"] = s["opcode_histogram"]
        metrics["sass_normalized_hash"] = s["normalized_hash"]
        metrics["sass_opcode_order_hash"] = s["opcode_order_hash"]
        metrics["sass_branch_count"] = s["branch_count"]
        metrics["sass_predicated_count"] = s["predicated_count"]
        metrics["sass_load_count"] = s["load_count"]
        metrics["sass_store_count"] = s["store_count"]

    if cubin_path.exists():
        metrics["cubin_bytes"] = cubin_path.stat().st_size
    if exe_path.exists():
        metrics["executable_bytes"] = exe_path.stat().st_size

    if compile_json.exists():
        try:
            cj = json.loads(compile_json.read_text())
            metrics["registers"] = cj.get("registers")
            v = cj.get("ptxas_verbose", "")
            metrics.update(disasm.parse_ptxas_verbose(v))
        except Exception:  # noqa: BLE001
            pass
    return metrics


def build_key(tc_id: str, profile_name: str) -> str:
    return f"{tc_id}__{profile_name}"
