"""GPU execution: invoke the native Driver API runner, parse result blobs.

Fixed-ABI and legacy standalone cases retain fresh-process isolation.  generator_ir
checksum cases may reuse one Driver API context for a bounded number of successful
jobs; the worker is discarded after any failure.  Per-device serialization is
provided by DeviceGate so one GPU only executes one kernel at a time.
"""

from __future__ import annotations

import atexit
import json
import re
import select
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
    if case.execution_mode == "driver_checksum":
        return _run_driver_checksum(
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


class _PersistentDriverRunner:
    """One long-lived native Driver API process/context for one GPU."""

    def __init__(self, executable: Path, device: int) -> None:
        self.executable = executable
        self.device = int(device)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._sequence = 0
        self._jobs = 0

    @staticmethod
    def _readline(proc: subprocess.Popen, timeout: float) -> Optional[str]:
        if proc.stdout is None:
            return None
        ready, _, _ = select.select([proc.stdout.fileno()], [], [], max(0.0, timeout))
        if not ready:
            return None
        line = proc.stdout.readline()
        return line.rstrip("\r\n") if line else None

    def _start(self, timeout: float) -> None:
        self._proc = subprocess.Popen(
            [str(self.executable), "--server", "--device", str(self.device)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        line = self._readline(self._proc, max(30.0, timeout))
        if line != "READY":
            detail = ""
            if self._proc.poll() is not None and self._proc.stderr is not None:
                detail = self._proc.stderr.read()[-500:]
            self._stop(force=True)
            raise RuntimeError(f"persistent CUDA runner failed to start: {line!r} {detail}")
        self._jobs = 0

    def _stop(self, force: bool = False) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None and not force and proc.stdin is not None:
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        with self._lock:
            self._stop()

    def run(self, spec_path: Path, timeout: float, restart_after: int) -> int:
        with self._lock:
            if self._proc is not None and self._proc.poll() is not None:
                self._stop(force=True)
            if self._proc is not None and restart_after > 0 and self._jobs >= restart_after:
                self._stop()
            if self._proc is None:
                self._start(timeout)
            assert self._proc is not None and self._proc.stdin is not None
            self._sequence += 1
            sequence = str(self._sequence)
            try:
                self._proc.stdin.write(f"RUN\t{sequence}\t{spec_path}\n")
                self._proc.stdin.flush()
            except OSError as exc:
                self._stop(force=True)
                raise RuntimeError(f"persistent CUDA runner write failed: {exc}") from exc
            line = self._readline(self._proc, timeout)
            if line is None:
                self._stop(force=True)
                raise subprocess.TimeoutExpired(str(self.executable), timeout)
            fields = line.split("\t")
            if len(fields) != 3 or fields[0] != "DONE" or fields[1] != sequence:
                self._stop(force=True)
                raise RuntimeError(f"invalid persistent CUDA runner response: {line!r}")
            try:
                status = int(fields[2])
            except ValueError as exc:
                self._stop(force=True)
                raise RuntimeError(f"invalid persistent CUDA runner status: {line!r}") from exc
            self._jobs += 1
            if status != EXIT_OK:
                # The native server exits after any failing job; discard the
                # potentially poisoned context even if it has not exited yet.
                self._stop(force=True)
            return status


_PERSISTENT_RUNNERS: Dict[tuple, _PersistentDriverRunner] = {}
_PERSISTENT_RUNNERS_LOCK = threading.Lock()


def _persistent_runner(executable: Path, device: int) -> _PersistentDriverRunner:
    key = (str(executable.resolve()), int(device))
    with _PERSISTENT_RUNNERS_LOCK:
        runner = _PERSISTENT_RUNNERS.get(key)
        if runner is None:
            runner = _PersistentDriverRunner(executable, device)
            _PERSISTENT_RUNNERS[key] = runner
        return runner


def _close_persistent_runners() -> None:
    with _PERSISTENT_RUNNERS_LOCK:
        runners = list(_PERSISTENT_RUNNERS.values())
        _PERSISTENT_RUNNERS.clear()
    for runner in runners:
        runner.close()


atexit.register(_close_persistent_runners)


def _clean_spec_value(value: Any, field: str) -> str:
    text = str(value)
    if "\n" in text or "\r" in text:
        raise ValueError(f"{field} contains a newline")
    return text


def _write_driver_job_spec(config: Config, case: GeneratedCase, build_dir: Path,
                           run_index: Any, input_salt: int,
                           timeout_ms: int = 15000) -> Path:
    spec = case.execution_spec
    if not isinstance(spec, dict) or spec.get("kind") != "generator_ir_checksum":
        raise ValueError("generator_ir Driver execution_spec is missing or unsupported")
    required = ("kernel", "args", "grid", "block", "n", "buf_size", "in_size")
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError(f"generator_ir Driver execution_spec missing {missing}")

    out_path = build_dir / f"output_run_{run_index}.bin"
    result_path = build_dir / f"execution_run_{run_index}.json"
    lines = [
        "version=1",
        f"cubin={_clean_spec_value(build_dir / 'kernel.cubin', 'cubin')}",
        f"kernel={_clean_spec_value(spec['kernel'], 'kernel')}",
        f"output={_clean_spec_value(out_path, 'output')}",
        f"result={_clean_spec_value(result_path, 'result')}",
        f"test_id={_clean_spec_value(case.case_id(), 'test_id')}",
        "grid=" + ",".join(str(int(x)) for x in spec["grid"]),
        "block=" + ",".join(str(int(x)) for x in spec["block"]),
        f"shared_bytes={int(spec.get('shared_bytes', 0))}",
        f"n={int(spec['n'])}",
        f"buf_size={int(spec['buf_size'])}",
        f"in_size={int(spec['in_size'])}",
        f"input_dist={_clean_spec_value(spec.get('input_distribution', 'hash'), 'input_dist')}",
        f"input_salt={int(input_salt) & 0xFFFFFFFF}",
        f"guards={int(config.execution.get('device_memory_guard_words', 8))}",
        f"timeout_ms={int(timeout_ms)}",
    ]
    for arg in spec["args"]:
        kind = str(arg.get("kind", ""))
        if kind == "scalar":
            payload = str(arg.get("data_hex", ""))
            if not re.fullmatch(r"[0-9a-fA-F]+", payload) or len(payload) % 2:
                raise ValueError(f"invalid scalar parameter encoding for {arg.get('name')}")
            lines.append(f"arg=scalar:{payload}")
        elif kind == "output_fp":
            lines.append(f"arg=output_fp:{int(arg['element_bytes'])}")
        elif kind in {"output", "input", "n"}:
            lines.append(f"arg={kind}")
        else:
            raise ValueError(f"unsupported generator_ir Driver arg kind {kind!r}")
    for global_spec in spec.get("globals", []):
        init = _clean_spec_value(global_spec.get("init", ""), "global init")
        name = _clean_spec_value(global_spec.get("name", ""), "global name")
        if init not in {"zero", "cmem_u32"} or not name:
            raise ValueError(f"unsupported generator_ir global spec {global_spec!r}")
        lines.append(f"global={init}:{name}")

    path = build_dir / f"driver_job_{run_index}.spec"
    path.write_text("\n".join(lines) + "\n")
    return path


def _run_driver_checksum(config: Config, tc: Toolchain,
                         case: GeneratedCase, build_dir: Path,
                         run_index: Any, device: int,
                         timeout: Optional[float], input_salt: int) -> RunResult:
    """Run a generator_ir cubin through its recorded dynamic Driver ABI."""
    runner = runner_path(config)
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin = build_dir / "kernel.cubin"
    out_path = build_dir / f"output_run_{run_index}.bin"
    exec_path = build_dir / f"execution_run_{run_index}.json"
    out_path.unlink(missing_ok=True)
    exec_path.unlink(missing_ok=True)
    if not cubin.exists():
        return RunResult(status="BAD_SPEC", output_words=None, error="cubin missing")
    limit = float(timeout or config.execution.get("run_timeout", 60))
    try:
        spec_path = _write_driver_job_spec(
            config, case, build_dir, run_index, input_salt,
            timeout_ms=int(limit * 1000))
    except (KeyError, TypeError, ValueError) as exc:
        info = {"status": "BAD_SPEC", "returncode": EXIT_BAD_SPEC, "error": str(exc)}
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="BAD_SPEC", output_words=None, error=str(exc), execution=info)

    restart_after = int(config.execution.get("runner_restart_after", 0))
    try:
        if restart_after > 0:
            returncode = _persistent_runner(runner, device).run(
                spec_path, limit, restart_after)
        else:
            proc = subprocess.run(
                [str(runner), "--ir-spec", str(spec_path), "--device", str(device)],
                capture_output=True, text=True, timeout=limit)
            returncode = proc.returncode
    except subprocess.TimeoutExpired:
        info = {"status": "TIMEOUT", "returncode": EXIT_TIMEOUT,
                "elapsed_us": int(limit * 1_000_000),
                "error": f"driver checksum runner timed out after {limit}s"}
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="TIMEOUT", output_words=None, error=info["error"],
                         elapsed_us=info["elapsed_us"], execution=info)
    except (OSError, RuntimeError) as exc:
        info = {"status": "CUDA_ERROR", "returncode": EXIT_CUDA_ERROR,
                "error": str(exc)}
        exec_path.write_text(json.dumps(info, indent=2))
        return RunResult(status="CUDA_ERROR", output_words=None,
                         error=str(exc), execution=info)
    finally:
        spec_path.unlink(missing_ok=True)

    exec_info: Dict[str, Any] = {}
    if exec_path.exists():
        try:
            exec_info = json.loads(exec_path.read_text())
        except Exception:  # noqa: BLE001
            exec_info = {"status": "PARSE_ERR"}
    if returncode != EXIT_OK and not out_path.exists():
        status = {EXIT_TIMEOUT: "TIMEOUT", EXIT_OOB: "OUT_OF_BOUNDS_WRITE",
                  EXIT_BAD_SPEC: "BAD_SPEC"}.get(returncode, "CUDA_ERROR")
        return RunResult(status=status, output_words=None,
                         error=exec_info.get("error", f"runner exited {returncode}"),
                         elapsed_us=int(exec_info.get("elapsed_us", 0)),
                         execution=exec_info)
    try:
        result = parse_output_blob(out_path.read_bytes())
    except OSError as exc:
        return RunResult(status="CUDA_ERROR", output_words=None, error=str(exc),
                         execution=exec_info)
    result.execution = exec_info
    result.elapsed_us = int(exec_info.get("elapsed_us", 0))
    result.error = "" if result.status == "OK" else exec_info.get("error", "")
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
                "device": int(device),
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
    # For the unsalted fast pass, ask the harness to also emit a salt=1 checksum
    # in the same process (reusing the CUDA context), so the input-sensitivity
    # oracle does not need a second process launch.
    also_salt = int(input_salt) == 0
    if also_salt:
        env["CUDA_STRESS_ALSO_SALT"] = "1"
    t0 = time.monotonic()
    try:
        proc = subprocess.run([str(exe)], capture_output=True, text=True,
                              timeout=limit, env=env, cwd=build_dir)
    except subprocess.TimeoutExpired as exc:
        info = {
            "status": "TIMEOUT", "returncode": 124,
            "elapsed_us": int(limit * 1_000_000),
            "device": int(device),
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
                "device": int(device),
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
    salted_checksum: Optional[int] = None
    if proc.returncode != 0:
        status = "CUDA_ERROR"
        error = proc.stderr[-500:] or f"standalone harness exited {proc.returncode}"
    elif not matches:
        status = "BAD_SPEC"
        error = "expected at least one checksum=<u64> line, found none"
    elif any(not 0 <= m <= 0xFFFFFFFFFFFFFFFF for m in matches):
        status = "BAD_SPEC"
        error = f"checksum outside u64 range: {matches}"
    else:
        checksum = matches[0]
        if len(matches) >= 2:
            salted_checksum = matches[1]

    info = {
        "status": status,
        "returncode": proc.returncode,
        "elapsed_us": elapsed_us,
        "checksum": checksum,
        "salted_checksum": salted_checksum,
        "device": int(device),
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
    if salted_checksum is not None:
        salt_path = build_dir / "output_run_input_sensitivity.bin"
        _write_ok_blob(salt_path, case.case_id(), struct.pack("<Q", salted_checksum))
    return RunResult(status="OK", output_words=payload, meaningful_words=1,
                     elapsed_us=elapsed_us, execution=info)


class DeviceGate:
    """Gate GPU work per device.

    By default (``workers_per_device == 1``) one kernel runs on a GPU at a time,
    which keeps runs isolated and deterministic.  Raising it lets several cases
    run concurrently on the same GPU: their host<->device copies (DMA) overlap
    with each other's kernel execution, though the kernels themselves only truly
    co-schedule under CUDA MPS (``nvidia-cuda-mps-control``), not plain streams.
    """

    def __init__(self, gpu_devices: List[int], workers_per_device: int = 1) -> None:
        self._workers = max(1, int(workers_per_device))
        self._gates: Dict[int, threading.BoundedSemaphore] = {
            d: threading.BoundedSemaphore(self._workers) for d in gpu_devices
        }

    def acquire(self, device: int):
        gate = self._gates.setdefault(device, threading.BoundedSemaphore(self._workers))
        gate.acquire()
        return _DeviceLease(gate)

    def __call__(self, device: int):
        return self.acquire(device)


class _DeviceLease:
    def __init__(self, gate) -> None:
        self._gate = gate

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._gate.release()
