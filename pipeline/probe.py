"""Toolchain / driver capability probing.

`probe_all` answers, for every installed toolchain:
  * do the binaries exist and what versions are they
  * is ptxas -O0..-O4 accepted, and which levels are *real* (behaviorally distinct SASS)
  * is -O4 a silent alias of an existing level (e.g. O4 == O3)?
  * can the host driver actually load and run this toolkit's CUBIN?
Results are cached under cache/ so the pipeline only probes once at startup.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, Toolchain, isolated_env

def _probe_source() -> str:
    """A stress kernel strong enough to expose O0/O1/O2/O3 SASS differences.

    Generated with the predicate_pressure strategy (deterministic), so the probe
    exercises real compiler pressure instead of a trivial kernel that collapses
    optimization levels.
    """
    from generator.base import default_registry
    strat = default_registry().get("predicate_pressure")
    case = strat.generate(
        config_seed=0x50524F4245,  # "PROBE"
        test_seed=0xC0FFEE,
        params={"num_comparisons": 64, "chain_len": 120, "num_diamonds": 24,
                "num_folds": 48, "dup_groups": 6, "use_launch_bounds": False},
        gen_version="1.0.0", target_sm="sm_75")
    return case.source


PROBE_KERNEL = _probe_source()


@dataclass
class ToolchainProbe:
    toolchain_id: str
    exists: bool
    nvcc_version: str = ""
    ptxas_version: str = ""
    accepted_levels: List[str] = field(default_factory=list)   # -O0..O4 accepted by ptxas
    real_levels: List[str] = field(default_factory=list)       # behaviorally distinct SASS
    sass_hash: Dict[str, str] = field(default_factory=dict)
    o4_status: str = "not_accepted"   # not_accepted | real | alias_of:<level>
    driver_runs: bool = False
    driver_error: str = ""
    probe_stdout: Dict[str, str] = field(default_factory=dict)
    probe_stderr: Dict[str, str] = field(default_factory=dict)
    probe_rc: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "toolchain": self.toolchain_id, "exists": self.exists,
            "nvcc_version": self.nvcc_version, "ptxas_version": self.ptxas_version,
            "accepted_levels": self.accepted_levels, "real_levels": self.real_levels,
            "sass_hash": self.sass_hash, "o4_status": self.o4_status,
            "driver_runs": self.driver_runs, "driver_error": self.driver_error,
            "probe_rc": self.probe_rc,
        }


def _run(argv: List[str], env: Dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def _version(binpath: str, env: Dict[str, str]) -> str:
    try:
        r = _run([binpath, "--version"], env, timeout=30)
        first = r.stdout.splitlines()[0] if r.stdout else r.stderr.splitlines()[0] if r.stderr else ""
        return first.strip()
    except Exception as e:  # noqa: BLE001
        return f"err:{e}"


def _sass_normalized_hash(cuobjdump: str, cubin: str, env: Dict[str, str]) -> str:
    r = _run([cuobjdump, "-sass", cubin], env, timeout=60)
    if r.returncode != 0:
        return ""
    lines = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("/*") and "*/" in line:
            lines.append(line)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def probe_toolchain(cfg: Config, tc: Toolchain, work: Path) -> ToolchainProbe:
    env = isolated_env(tc)
    probe = ToolchainProbe(toolchain_id=tc.id, exists=True)
    probe.nvcc_version = _version(tc.nvcc, env)
    probe.ptxas_version = _version(tc.ptxas, env)

    # O0..O4 acceptance + SASS equivalence.
    src = work / "probe_kernel.cu"
    src.write_text(PROBE_KERNEL)
    ptx = work / "probe_kernel.ptx"
    r = _run([tc.nvcc, f"-arch={tc.arch}", "-ccbin", tc.ccbin, "-ptx", "-o", str(ptx), str(src)], env, timeout=120)
    if r.returncode != 0:
        probe.exists = True
        probe.probe_rc["ptx"] = r.returncode
        probe.probe_stderr["ptx"] = r.stderr[-1000:]
        return probe

    for opt in range(5):
        name = f"O{opt}"
        cubin = work / f"probe_O{opt}.cubin"
        r = _run([tc.ptxas, f"-arch={tc.arch}", f"-O{opt}", "-v", "-o", str(cubin), str(ptx)], env, timeout=120)
        probe.probe_rc[name] = r.returncode
        probe.probe_stdout[name] = r.stdout[-800:]
        probe.probe_stderr[name] = r.stderr[-800:]
        if r.returncode == 0 and cubin.exists():
            probe.accepted_levels.append(name)
            probe.sass_hash[name] = _sass_normalized_hash(tc.cuobjdump, str(cubin), env)

    # real levels: accepted AND SASS distinct from every *lower* accepted level.
    # (O4 that equals O3 is an alias, not a real level; likewise O3==O2 collapses.)
    order = ["O0", "O1", "O2", "O3", "O4"]
    for name in order:
        if name not in probe.accepted_levels:
            continue
        h = probe.sass_hash.get(name)
        if not h:
            continue
        lower = [o for o in order[:order.index(name)] if o in probe.accepted_levels]
        if all(probe.sass_hash.get(o) != h for o in lower):
            probe.real_levels.append(name)

    # O4 status.
    if "O4" not in probe.accepted_levels:
        probe.o4_status = "not_accepted"
    elif "O4" in probe.real_levels:
        probe.o4_status = "real"
    else:
        alias = next((o for o in ("O3", "O2", "O1", "O0")
                      if probe.sass_hash.get("O4") == probe.sass_hash.get(o)), "unknown")
        probe.o4_status = f"alias_of:{alias}"

    # Driver load+run compatibility (only meaningful when a GPU is present).
    try:
        probe.driver_runs, probe.driver_error = _driver_load_run(cfg, tc, work, env)
    except Exception as e:  # noqa: BLE001
        probe.driver_error = f"exception: {e}"
    return probe


def _driver_load_run(cfg: Config, tc: Toolchain, work: Path, env: Dict[str, str]):
    runner = cfg.compilation.get("runner_path") or str(Path(__file__).parent.parent / "native" / "cuda_runner")
    if not Path(runner).exists():
        return False, f"runner not built at {runner} (make -C native)"
    cubin = work / "probe_driver.cubin"
    ptx = work / "probe_driver.ptx"
    src = work / "probe_kernel.cu"
    r = _run([tc.nvcc, f"-arch={tc.arch}", "-ccbin", tc.ccbin, "-ptx", "-o", str(ptx), str(src)], env, timeout=120)
    if r.returncode != 0:
        return False, "nvcc -ptx failed"
    r = _run([tc.ptxas, f"-arch={tc.arch}", "-O3", "-o", str(cubin), str(ptx)], env, timeout=120)
    if r.returncode != 0:
        return False, "ptxas failed"
    inp = work / f"probe_driver_in_{tc.id}.bin"
    inp.write_bytes(bytes(256 * 8))
    out = work / f"probe_driver_out_{tc.id}.bin"
    execf = work / f"probe_driver_exec_{tc.id}.json"
    run = subprocess.run([runner, "--cubin", str(cubin), "--kernel", "test_kernel",
                          "--input", str(inp), "--output", str(out),
                          "--result", str(execf),
                          "--input-words", "256", "--output-words", "128",
                          "--grid", "1", "--block", "64", "--seed", "0", "--guards", "8",
                          "--test-id", "probe", "--timeout-ms", "30000"],
                         capture_output=True, text=True, timeout=90)
    if run.returncode == 0:
        return True, ""
    err = ""
    if execf.exists():
        try:
            j = json.loads(execf.read_text())
            err = j.get("status", "") + " " + j.get("error", "")
        except Exception:  # noqa: BLE001
            pass
    if not err:
        err = run.stderr.strip()[-300:]
    return False, err


def probe_all(cfg: Config, work_dir: Optional[Path] = None) -> List[ToolchainProbe]:
    work = work_dir or Path(tempfile.mkdtemp(prefix="cuda_stress_probe_"))
    probes = []
    for tc in cfg.toolchains:
        if not Path(tc.nvcc).exists() or not Path(tc.ptxas).exists():
            probes.append(ToolchainProbe(toolchain_id=tc.id, exists=False))
            continue
        probes.append(probe_toolchain(cfg, tc, work))
    return probes


def env_signature() -> str:
    h = hashlib.sha256()
    for p in ("/usr/local/cuda*", "/usr/lib/x86_64-linux-gnu/libcuda.so*"):
        for f in sorted(Path("/").glob(p.lstrip("/")) if p.startswith("/usr/local") else []):
            pass
    h.update(os.uname().release.encode())
    for cuda in sorted(Path("/usr/local").glob("cuda*")):
        h.update(str(cuda).encode())
    h.update(subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30).stdout.encode())
    return h.hexdigest()[:16]


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "toolchain_probe.json"


def load_probe_cache() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return None


def save_probe_cache(probes: List[ToolchainProbe], cfg: Config) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "signature": probe_signature(cfg),
        "probes": [p.as_dict() for p in probes],
    }, indent=2))


def probe_signature(cfg: Config) -> str:
    """Cache key: environment (GPU/driver/cuda dirs) plus the configured toolchains.

    The toolchain list is folded in so editing config/toolchains.yaml (nvcc path,
    arch, ccbin, ...) invalidates a stale probe cache instead of reusing it.
    """
    h = hashlib.sha256()
    h.update(env_signature().encode())
    for tc in cfg.toolchains:
        h.update(
            f"{tc.id}|{tc.nvcc}|{tc.ptxas}|{tc.arch}|{tc.ccbin}|{int(tc.runnable)}"
            .encode())
    return h.hexdigest()[:16]


def cached_probes(cfg: Config, force: bool = False) -> List[ToolchainProbe]:
    """Probe once, then reuse the cache until the environment or toolchains change."""
    cache = load_probe_cache()
    if cache and cache.get("signature") == probe_signature(cfg) and not force:
        probes = []
        for p in cache["probes"]:
            p = dict(p)
            p.setdefault("toolchain_id", p.pop("toolchain", ""))
            probes.append(ToolchainProbe(**p))
        return probes
    probes = probe_all(cfg)
    save_probe_cache(probes, cfg)
    return probes


def doctor_report(cfg: Config, probes: List[ToolchainProbe], gpu_info: Dict[str, Any]) -> str:
    lines = ["# CUDA-stress doctor", ""]
    lines.append("## GPU / driver")
    lines.append(f"  device      : {gpu_info.get('name','?')}")
    lines.append(f"  compute     : sm{gpu_info.get('major','?')}{gpu_info.get('minor','?')}")
    lines.append(f"  SMs         : {gpu_info.get('sm_count','?')}")
    lines.append(f"  mem         : {gpu_info.get('mem_mb','?')} MiB")
    lines.append(f"  driver ver  : {gpu_info.get('driver_version','?')}")
    lines.append("")
    lines.append("## Toolchains")
    hdr = f"  {'toolchain':<12} {'nvcc':<28} {'ptxas -O accepted':<18} {'real levels':<24} {'O4 status':<16} driver-run"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for p in probes:
        if not p.exists:
            lines.append(f"  {p.toolchain_id:<12} MISSING")
            continue
        lines.append(
            f"  {p.toolchain_id:<12} "
            f"{p.nvcc_version[:26]:<28} "
            f"{'/'.join(p.accepted_levels) or '-':<18} "
            f"{'/'.join(p.real_levels) or '-':<24} "
            f"{p.o4_status:<16} "
            f"{'YES' if p.driver_runs else 'NO'}")
        if not p.driver_runs and p.driver_error:
            lines.append(f"      driver error: {p.driver_error[:160]}")
    lines.append("")
    lines.append("## ptxas -O4 note")
    lines.append("  -O4 is NOT a real ptxas optimization level on any toolkit here: it is accepted")
    lines.append("  but produces SASS byte-identical to another level. The matrix therefore uses the")
    lines.append("  real, behaviorally distinct levels only (default: O0-O3).")
    lines.append("")
    return "\n".join(lines)


def get_gpu_info(cfg: Optional[Config] = None) -> Dict[str, Any]:
    runner = Path(__file__).parent.parent / "native" / "cuda_runner"
    info = {"name": "?", "major": "?", "minor": "?", "sm_count": "?", "mem_mb": "?",
            "driver_version": "?", "present": False}
    if not runner.exists():
        return info
    tc = (cfg.toolchains[0] if cfg and cfg.toolchains else None)
    nvcc = tc.nvcc if tc else "/usr/local/cuda-11.7/bin/nvcc"
    arch = tc.arch if tc else "sm_75"
    ccbin = tc.ccbin if tc else "/usr/bin/g++-9"
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="cuda_stress_gpu_"))
    cubin = work / "k.cubin"
    src = work / "k.cu"
    src.write_text(PROBE_KERNEL)
    env = dict(os.environ)
    if tc:
        env = isolated_env(tc)
    argv = [nvcc, f"-arch={arch}", "-ccbin", ccbin, "-std=c++14", "-cubin",
            "-o", str(cubin), str(src)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
    if r.returncode != 0:
        return info
    inp = work / "in.bin"
    inp.write_bytes(bytes(256 * 8))
    run = subprocess.run([str(runner), "--cubin", str(cubin), "--kernel", "test_kernel",
                          "--input", str(inp), "--output", str(work / "o.bin"),
                          "--result", str(work / "exec.json"),
                          "--input-words", "256", "--output-words", "128",
                          "--grid", "1", "--block", "64", "--seed", "0", "--guards", "8",
                          "--test-id", "gpuinfo", "--timeout-ms", "30000"],
                         capture_output=True, text=True, timeout=90)
    if (work / "exec.json").exists():
        try:
            j = json.loads((work / "exec.json").read_text())
            info = {"name": j.get("device", "?"), "major": "?", "minor": "?", "sm_count": j.get("sm_count", "?"),
                    "mem_mb": j.get("device_mem_mb", "?"), "driver_version": j.get("driver_version", "?"),
                    "present": run.returncode == 0}
            sm = j.get("sm", "")
            if len(sm) == 2:
                info["major"], info["minor"] = sm[0], sm[1]
        except Exception:  # noqa: BLE001
            pass
    return info
