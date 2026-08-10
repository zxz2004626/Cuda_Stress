"""GPU execution: invoke the native Driver API runner, parse result blobs.

Each run uses a fresh runner process => a poisoned CUDA context never contaminates the
next task.  Per-device serialization is provided by DeviceGate so one GPU only executes
one kernel at a time (determinism / no resource contention).
"""

from __future__ import annotations

import json
import struct
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, Toolchain
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
              run_index: int, device: int = 0, timeout: Optional[float] = None) -> RunResult:
    """Run one build once on `device`, write output_run_<N>.bin + execution_run_<N>.json."""
    runner = runner_path(config)
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin = build_dir / "kernel.cubin"
    if not cubin.exists():
        return RunResult(status="BAD_SPEC", output_words=None, error="cubin missing")
    out_path = build_dir / f"output_run_{run_index}.bin"
    exec_path = build_dir / f"execution_run_{run_index}.json"

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
        return RunResult(status="TIMEOUT", output_words=None,
                         error=f"runner timed out after {timeout or config.execution.get('run_timeout', 60)}s")

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
