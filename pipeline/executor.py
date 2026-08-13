"""GPU execution: invoke the native Driver API runner, parse result blobs.

Each run uses a fresh runner process => a poisoned CUDA context never contaminates the
next task.  Per-device serialization is provided by DeviceGate so one GPU only executes
one kernel at a time (determinism / no resource contention).
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, Toolchain, isolated_env
from generator.base import GeneratedCase

BLOB_HEADER = struct.Struct("<QQIIQQ")  # magic, test_id_hash, status, reserved, words, canary
BLOB_MAGIC = 0x4355434553545245
BLOB_CANARY = 0xDEADBEEFCAFEBABE
BLOB_TAIL_CANARY = 0xCAFEBABEDEADBEEF

RUNNER_DEFAULT = Path(__file__).resolve().parent.parent / "native" / "cuda_runner"

EXIT_OK = 0
EXIT_CUDA_ERROR = 1
EXIT_OOB = 2
EXIT_BAD_SPEC = 3
EXIT_TIMEOUT = 124


@dataclass
class RunResult:
    status: str                      # OK | CUDA_ERROR | OUT_OF_BOUNDS_WRITE | TIMEOUT | BAD_SPEC
    output_words: Optional[bytes]    # raw meaningful u64 words (little-endian)
    meaningful_words: int = 0
    error: str = ""
    elapsed_us: int = 0
    execution: Dict[str, Any] = field(default_factory=dict)


def parse_output_blob(data: bytes) -> RunResult:
    if len(data) < BLOB_HEADER.size + 8:
        return RunResult(status="BAD_SPEC", output_words=None, error="blob too short")
    magic, _thash, status, _rsv, words, canary = BLOB_HEADER.unpack(data[:BLOB_HEADER.size])
    tail = struct.unpack("<Q", data[-8:])[0]
    if magic != BLOB_MAGIC or canary != BLOB_CANARY or tail != BLOB_TAIL_CANARY:
        return RunResult(status="BAD_SPEC", output_words=None,
                         error=f"blob header corrupt magic={magic:#x} canary={canary:#x} tail={tail:#x}")
    body = data[BLOB_HEADER.size:-8]
    if status != 0:
        name = {1: "CUDA_ERROR", 2: "OUT_OF_BOUNDS_WRITE", 3: "BAD_SPEC", 4: "TIMEOUT"}.get(status, f"STATUS_{status}")
        return RunResult(status=name, output_words=None, meaningful_words=words)
    if len(body) != words * 8:
        return RunResult(status="BAD_SPEC", output_words=None,
                         error=f"blob word count mismatch: header={words}, body={len(body)//8}")
    return RunResult(status="OK", output_words=body, meaningful_words=words)


def runner_path(config: Config) -> Path:
    p = Path(config.compilation.get("runner_path") or RUNNER_DEFAULT)
    if not p.exists():
        raise FileNotFoundError(f"runner not built at {p} (run `make -C native`)")
    return p


def run_build(config: Config, tc: Toolchain, case: GeneratedCase, build_dir: Path,
              run_index: Any, device: int = 0, timeout: Optional[float] = None,
              input_salt: int = 0) -> RunResult:
    """Run one build once on `device`, write output_run_<N>.bin + execution_run_<N>.json."""
    if case.execution_mode == "standalone_checksum":
        return _run_standalone_checksum(
            config, tc, case, build_dir, run_index, device, timeout, input_salt)

    runner = runner_path(config)
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin = build_dir / "kernel.cubin"
    out_path = build_dir / f"output_run_{run_index}.bin"
    exec_path = build_dir / f"execution_run_{run_index}.json"
    # Never let a failed re-run inherit an apparently valid result from an
    # earlier invocation with the same run tag.
    out_path.unlink(missing_ok=True)
    exec_path.unlink(missing_ok=True)
    if not cubin.exists():
        return RunResult(status="BAD_SPEC", output_words=None, error="cubin missing")

    grid = ",".join(str(x) for x in case.grid)
    block = ",".join(str(x) for x in case.block)
    total = case.total_threads()
    output_words = total * case.output_words_per_thread

    argv = [
        str(runner),
        "--cubin", str(cubin),
        "--kernel", case.kernel_entry,
        "--input", str(build_dir.parent.parent / "input.bin"),
        "--output", str(out_path),
        "--result", str(exec_path),
        "--input-words", str(case.input_words),
        "--output-words", str(output_words),
        "--grid", grid,
        "--block", block,
        "--seed", str(case.kernel_seed),
        "--guards", str(config.execution.get("device_memory_guard_words", 8)),
        "--test-id", case.case_id(),
        "--device", str(device),
        "--timeout-ms", str(int((timeout or config.execution.get("run_timeout", 60)) * 1000)),
    ]
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout or config.execution.get("run_timeout", 60))
    except subprocess.TimeoutExpired:
        limit = timeout or config.execution.get("run_timeout", 60)
        info = {"status": "TIMEOUT", "returncode": EXIT_TIMEOUT,
                "error": f"runner timed out after {limit}s"}
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="TIMEOUT", output_words=None,
                         error=info["error"], execution=info)

    exec_info: Dict[str, Any] = {}
    if exec_path.exists():
        try:
            exec_info = json.loads(exec_path.read_text())
        except Exception:  # noqa: BLE001
            exec_info = {"status": "PARSE_ERR", "stderr_tail": r.stderr[-500:]}

    if r.returncode != EXIT_OK and not out_path.exists():
        status = "TIMEOUT" if r.returncode == EXIT_TIMEOUT else "CUDA_ERROR"
        return RunResult(status=status, output_words=None,
                         error=exec_info.get("error") or r.stderr[-500:],
                         execution=exec_info)

    data = out_path.read_bytes()
    result = parse_output_blob(data)
    result.execution = exec_info
    result.elapsed_us = exec_info.get("elapsed_us", 0)
    if result.status == "OK":
        result.error = ""
    else:
        result.error = exec_info.get("error", "")
    return result


_CHECKSUM_RE = re.compile(r"^checksum=([0-9]+)$")


def _test_id_hash(test_id: str) -> int:
    """64-bit FNV-1a, matching the native runner's diagnostic id field."""
    h = 1469598103934665603
    for b in test_id.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _write_ok_blob(path: Path, test_id: str, payload: bytes) -> None:
    if len(payload) % 8:
        raise ValueError("result payload must contain whole u64 words")
    words = len(payload) // 8
    blob = BLOB_HEADER.pack(
        BLOB_MAGIC, _test_id_hash(test_id), 0, 0, words, BLOB_CANARY)
    path.write_bytes(blob + payload + struct.pack("<Q", BLOB_TAIL_CANARY))


def _run_standalone_checksum(config: Config, tc: Toolchain,
                             case: GeneratedCase, build_dir: Path,
                             run_index: Any, device: int,
                             timeout: Optional[float], input_salt: int) -> RunResult:
    """Execute a generator-owned harness and normalize its checksum to a blob."""
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "kernel.exe"
    out_path = build_dir / f"output_run_{run_index}.bin"
    exec_path = build_dir / f"execution_run_{run_index}.json"
    out_path.unlink(missing_ok=True)
    exec_path.unlink(missing_ok=True)
    if not exe.exists():
        info = {"status": "BAD_SPEC", "returncode": -1,
                "error": "standalone executable missing",
                "input_salt": int(input_salt) & 0xFFFFFFFF}
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="BAD_SPEC", output_words=None,
                         error=info["error"], execution=info)

    limit = float(timeout or config.execution.get("run_timeout", 60))
    env = isolated_env(tc)
    # The harness uses the runtime API and selects its default logical device.
    # CUDA_VISIBLE_DEVICES maps the requested physical device to logical 0.
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["CUDA_STRESS_INPUT_SALT"] = str(int(input_salt) & 0xFFFFFFFF)
    t0 = time.monotonic()
    try:
        proc = subprocess.run([str(exe)], capture_output=True, text=True,
                              timeout=limit, env=env, cwd=build_dir)
    except subprocess.TimeoutExpired as exc:
        info = {
            "status": "TIMEOUT", "returncode": 124,
            "elapsed_us": int(limit * 1_000_000),
            "input_salt": int(input_salt) & 0xFFFFFFFF,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
        }
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="TIMEOUT", output_words=None,
                         error=f"standalone harness timed out after {limit}s",
                         elapsed_us=info["elapsed_us"], execution=info)
    except OSError as exc:
        info = {"status": "BAD_SPEC", "returncode": -1, "error": str(exc),
                "input_salt": int(input_salt) & 0xFFFFFFFF}
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="BAD_SPEC", output_words=None,
                         error=str(exc), execution=info)

    elapsed_us = int((time.monotonic() - t0) * 1_000_000)
    matches = []
    for line in proc.stdout.splitlines():
        match = _CHECKSUM_RE.fullmatch(line.strip())
        if match:
            matches.append(int(match.group(1)))

    status = "OK"
    error = ""
    checksum: Optional[int] = None
    if proc.returncode != 0:
        status = "CUDA_ERROR"
        error = proc.stderr[-500:] or f"standalone harness exited {proc.returncode}"
    elif len(matches) != 1:
        status = "BAD_SPEC"
        error = f"expected exactly one checksum=<u64> line, found {len(matches)}"
    elif not 0 <= matches[0] <= 0xFFFFFFFFFFFFFFFF:
        status = "BAD_SPEC"
        error = f"checksum outside u64 range: {matches[0]}"
    else:
        checksum = matches[0]

    info = {
        "status": status,
        "returncode": proc.returncode,
        "elapsed_us": elapsed_us,
        "checksum": checksum,
        "input_salt": int(input_salt) & 0xFFFFFFFF,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
    }
    exec_path.write_text(json.dumps(info, indent=2))
    if status != "OK" or checksum is None:
        return RunResult(status=status, output_words=None, error=error,
                         elapsed_us=elapsed_us, execution=info)

    payload = struct.pack("<Q", checksum)
    _write_ok_blob(out_path, case.case_id(), payload)
    return RunResult(status="OK", output_words=payload, meaningful_words=1,
                     elapsed_us=elapsed_us, execution=info)


class DeviceGate:
    """Serialize GPU work per device.  One thread per device may run a kernel at a time."""

    def __init__(self, gpu_devices: List[int]) -> None:
        self._locks: Dict[int, threading.Lock] = {d: threading.Lock() for d in gpu_devices}

    def acquire(self, device: int):
        lock = self._locks.setdefault(device, threading.Lock())
        lock.acquire()
        return _DeviceLease(lock)

    def __call__(self, device: int):
        return self.acquire(device)


class _DeviceLease:
    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._lock.release()
