"""External profiler adapters: rocprofv3 (HIP) and nsys (CUDA)."""

from __future__ import annotations

import csv
import glob
import io
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional

log = logging.getLogger(__name__)


def profile_hip(exe_path: str, profiler: str = "rocprofv3",
                timeout: int = 120) -> Optional[float]:
    """Kernel time in microseconds from rocprofv3 --stats output."""
    if not os.path.isfile(exe_path):
        return None
    with tempfile.TemporaryDirectory(prefix="rocprof_") as tmpdir:
        out_csv = os.path.join(tmpdir, "results.csv")
        cmd = [profiler, "--kernel-trace", "--stats",
               "-o", out_csv, "-f", "csv", "--", exe_path]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "HIP_VISIBLE_DEVICES": "0",
                     "AMD_LOG_LEVEL": "0"})
        except (subprocess.TimeoutExpired, OSError):
            # OSError subsumes FileNotFoundError; also covers PermissionError
            # and fork/EMFILE failures so the function honors its
            # Optional[float] contract instead of propagating.
            return None
        if proc.returncode != 0:
            log.debug("rocprofv3 failed rc=%d: %s", proc.returncode,
                      proc.stderr[:300])
            return None

        candidates = glob.glob(os.path.join(tmpdir, "*kernel_stats*"))
        if not candidates and os.path.isfile(out_csv):
            candidates = [out_csv]
        if not candidates:
            return None
        try:
            content = open(candidates[0]).read()
        except OSError:
            return None

        total_ns = 0.0
        count = 0
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            name = row.get("Name", "")
            if name.startswith("__amd_rocclr"):
                continue
            try:
                total_ns += float(row.get("AverageNs", ""))
                count += 1
            except (ValueError, TypeError):
                continue
        if count == 0:
            return None
        return total_ns / 1000.0


_NSYS_ROW = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+"
    r"([\d.]+)\s*(us|ms|ns|s)\s+"
    r"([\d.]+)\s*(us|ms|ns|s)\s+"
    r"([\d.]+)\s*(us|ms|ns|s)\s+"
    r"([\d.]+)\s*(us|ms|ns|s)",
    re.MULTILINE,
)

_TO_US = {"ns": 0.001, "us": 1.0, "ms": 1000.0, "s": 1_000_000.0}


def profile_cuda(exe_path: str, profiler: str = "nsys",
                 timeout: int = 120) -> Optional[float]:
    """Kernel time in microseconds from nsys --stats output."""
    if not os.path.isfile(exe_path):
        return None
    with tempfile.TemporaryDirectory(prefix="nsys_") as tmpdir:
        report = os.path.join(tmpdir, "report")
        cmd = [profiler, "profile", "--stats=true",
               "--force-overwrite=true", "-o", report, exe_path]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        except (subprocess.TimeoutExpired, OSError):
            return None
        if proc.returncode != 0:
            log.debug("nsys failed rc=%d: %s", proc.returncode,
                      proc.stderr[:300])
            return None
        return _parse_nsys(proc.stdout + "\n" + proc.stderr)


def _parse_nsys(output: str) -> Optional[float]:
    total_us = 0.0
    total_calls = 0
    for m in _NSYS_ROW.finditer(output):
        calls = int(m.group(2))
        avg_val = float(m.group(3))
        avg_us = avg_val * _TO_US.get(m.group(4), 1.0)
        total_us += avg_us * calls
        total_calls += calls
    if total_calls > 0:
        return total_us / total_calls

    # Fallback: "GPU Kernel Statistics" section with an Avg (ns) column.
    in_section = False
    for line in output.splitlines():
        if "Kernel Statistics" in line:
            in_section = True
            continue
        if in_section and line.strip():
            parts = line.split()
            nums = [p.replace(",", "") for p in parts
                    if p.replace(".", "").replace(",", "").isdigit()]
            if len(nums) >= 4:
                try:
                    return float(nums[3]) / 1000.0
                except ValueError:
                    continue
    return None
