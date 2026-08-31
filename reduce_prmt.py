#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent
CASE = REPO / "corpus/differentials/bug_reports/inline_ptx__7e79400da6b65775"
SOURCE = CASE / "test.cu"
WORK = CASE / "reduce_work.cu"
FINAL = CASE / "reduced_auto.cu"
NVCC = Path("/usr/local/cuda-12.6/bin/nvcc")
RUNNER = REPO / "native/cuda_runner"
TARGET = '"prmt.b32 %%xr7, %%xr6, %%xr6, 0x89ab;\\n\\t"'

lines = SOURCE.read_text().splitlines(keepends=True)
body_start = next(i for i, line in enumerate(lines) if '"abs.s64 ' in line)
body_end = next(i for i, line in enumerate(lines) if '"mov.u64 %0, %%xd0;' in line)
target_index = next(i for i, line in enumerate(lines) if TARGET in line)
removable = [i for i in range(body_start, body_end) if i != target_index]
cache: dict[str, bool] = {}
tests = 0


def materialize(kept: list[int], path: Path) -> None:
    kept_set = set(kept)
    text = "".join(
        line
        for i, line in enumerate(lines)
        if i < body_start or i >= body_end or i == target_index or i in kept_set
    )
    path.write_text(text)


def run_quiet(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30
    )


def interesting(kept: list[int]) -> bool:
    global tests
    key = hashlib.sha256(",".join(map(str, kept)).encode()).hexdigest()
    if key in cache:
        return cache[key]
    tests += 1
    materialize(kept, WORK)
    cubins = [CASE / "reduce0.cubin", CASE / "reduce1.cubin"]
    compilers = [
        subprocess.Popen(
            [
                str(NVCC),
                "-arch=sm_86",
                "-cubin",
                f"-Xptxas=-O{level}",
                "-o",
                str(cubins[level]),
                str(WORK),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for level in (0, 1)
    ]
    if any(p.communicate(timeout=30)[0] or p.returncode for p in compilers):
        cache[key] = False
        return False
    outputs = [Path(f"/tmp/prmt_reduce{level}.bin") for level in (0, 1)]
    runners = [
        subprocess.Popen(
            [
                str(RUNNER),
                "--cubin",
                str(cubins[level]),
                "--kernel",
                "test_kernel",
                "--input",
                str(CASE / "input.bin"),
                "--output",
                str(outputs[level]),
                "--result",
                f"/tmp/prmt_reduce{level}.json",
                "--input-words",
                "256",
                "--output-words",
                "2",
                "--grid",
                "1",
                "--block",
                "1",
                "--seed",
                "15308625898037569020",
                "--guards",
                "8",
                "--test-id",
                "reduce",
                "--timeout-ms",
                "60000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for level in (0, 1)
    ]
    for process in runners:
        process.communicate(timeout=65)
    ok = all(process.returncode == 0 for process in runners)
    result = ok and outputs[0].read_bytes() != outputs[1].read_bytes()
    cache[key] = result
    return result


def ddmin(items: list[int]) -> list[int]:
    n = 2
    while len(items) >= 2:
        chunk = (len(items) + n - 1) // n
        reduced = False
        for start in range(0, len(items), chunk):
            candidate = items[:start] + items[start + chunk :]
            if interesting(candidate):
                items = candidate
                n = max(2, n - 1)
                reduced = True
                print(f"kept={len(items) + 1} tests={tests}", flush=True)
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
    return items


if not interesting(removable):
    raise SystemExit("full source did not reproduce O0/O1 divergence")

result = ddmin(removable)
materialize(result, FINAL)
print(f"done: {FINAL} has {len(result) + 1} body instructions after {tests} tests")
