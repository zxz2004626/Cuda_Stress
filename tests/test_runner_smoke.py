"""GPU integration smoke test: compile a real kernel, run it through cuda_runner,
verify the output blob (magic/canary/guards) and the CPU oracle match."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from generator.base import default_registry

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "native" / "cuda_runner"
NVCC = shutil.which("nvcc") or "/usr/local/cuda-12.6/bin/nvcc"


def _gpu_available():
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _gpu_available() or not RUNNER.exists(),
                                reason="no GPU or runner not built")


def test_runner_blob_and_oracle(tmp_path):
    reg = default_registry()
    case = reg.get("integer_chain").generate(
        20260810, 11, {"chain_len": 64, "use_mulhi": True, "use_funnelshift": True,
                       "use_bfe_bfi": True, "use_prmt": True, "use_lop3": True,
                       "boundary_words": 8}, "1.0.0", "sm_86")
    src = tmp_path / "test.cu"
    src.write_text(case.source)
    (tmp_path / "input.bin").write_bytes(case.input_bytes)
    ptx = tmp_path / "k.ptx"
    r = subprocess.run([NVCC, "-arch=sm_86", "-ptx", "-o", str(ptx), str(src)],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    cubin = tmp_path / "k.cubin"
    r = subprocess.run([NVCC, "-arch=sm_86", "-cubin", "-Xptxas=-O3",
                        "-o", str(cubin), str(src)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr

    out = tmp_path / "out.bin"
    execf = tmp_path / "exec.json"
    outw = case.total_threads() * case.output_words_per_thread
    run = subprocess.run([str(RUNNER), "--cubin", str(cubin), "--kernel", "test_kernel",
                          "--input", str(tmp_path / "input.bin"), "--output", str(out),
                          "--result", str(execf),
                          "--input-words", str(case.input_words),
                          "--output-words", str(outw),
                          "--grid", "1", "--block", "64",
                          "--seed", str(case.kernel_seed), "--guards", "8",
                          "--test-id", case.case_id(), "--timeout-ms", "60000"],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    data = out.read_bytes()
    magic, _h, status, _rsv, words, canary = struct.unpack("<QQIIQQ", data[:40])
    tail = struct.unpack("<Q", data[-8:])[0]
    assert magic == 0x4355434553545245
    assert canary == 0xDEADBEEFCAFEBABE
    assert tail == 0xCAFEBABEDEADBEEF
    assert status == 0
    assert words == outw
    assert data[40:-8] == case.expected_bytes
    j = json.loads(execf.read_text())
    assert j["status"] == "OK"
    assert j["guard_ok"] is True


def test_inline_ptx_roundtrip(tmp_path):
    """Random inline-PTX kernel must compile, run, and match its CPU oracle exactly."""
    from generator.strategies.inline_ptx import InlinePtx
    reg = default_registry()
    strat = reg.get("inline_ptx")
    case = strat.generate(
        20260810, 4242,
        {"n_instr": 80, "n_regs32": 8, "n_regs64": 5, "n_preds": 4,
         "use_rare_ops": True, "use_immediates": True}, "1.0.0", "sm_86")
    src = tmp_path / "test.cu"
    src.write_text(case.source)
    (tmp_path / "input.bin").write_bytes(case.input_bytes)
    cubin = tmp_path / "k.cubin"
    r = subprocess.run([NVCC, "-arch=sm_86", "-cubin", "-Xptxas=-O2",
                        "-o", str(cubin), str(src)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out.bin"
    execf = tmp_path / "exec.json"
    outw = case.total_threads() * case.output_words_per_thread
    run = subprocess.run([str(RUNNER), "--cubin", str(cubin), "--kernel", "test_kernel",
                          "--input", str(tmp_path / "input.bin"), "--output", str(out),
                          "--result", str(execf),
                          "--input-words", str(case.input_words),
                          "--output-words", str(outw),
                          "--grid", "1", "--block", "64",
                          "--seed", str(case.kernel_seed), "--guards", "8",
                          "--test-id", case.case_id(), "--timeout-ms", "60000"],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    data = out.read_bytes()
    status = struct.unpack("<I", data[16:20])[0]
    assert status == 0, json.loads(execf.read_text())
    assert data[40:-8] == case.expected_bytes


def test_runner_detects_oob(tmp_path):
    """A kernel that writes just past the output buffer must trip the guard -> exit 2."""
    src = tmp_path / "oob.cu"
    src.write_text(
        'extern "C" __global__ void test_kernel(const unsigned long long* i,'
        ' unsigned long long* o, unsigned int n, unsigned long long s) {\n'
        '  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;\n'
        '  o[tid + 100u] = s ^ tid;\n'   # beyond the 128-word meaningful region + guards
        '}\n')
    ptx = tmp_path / "oob.ptx"
    subprocess.run([NVCC, "-arch=sm_86", "-ptx", "-o", str(ptx), str(src)],
                   capture_output=True, timeout=180, check=True)
    cubin = tmp_path / "oob.cubin"
    subprocess.run([NVCC, "-arch=sm_86", "-cubin", "-Xptxas=-O3", "-o", str(cubin), str(src)],
                   capture_output=True, timeout=180, check=True)
    (tmp_path / "in.bin").write_bytes(bytes(256 * 8))
    run = subprocess.run([str(RUNNER), "--cubin", str(cubin), "--kernel", "test_kernel",
                          "--input", str(tmp_path / "in.bin"),
                          "--output", str(tmp_path / "o.bin"),
                          "--result", str(tmp_path / "e.json"),
                          "--input-words", "256", "--output-words", "128",
                          "--grid", "1", "--block", "64", "--seed", "0", "--guards", "8",
                          "--test-id", "oob", "--timeout-ms", "30000"],
                         capture_output=True, text=True, timeout=120)
    # Either the launch faults (CUDA_ERROR) or the guard catches it (exit 2). Both prove
    # the runner does not silently return garbage.
    assert run.returncode in (1, 2)
