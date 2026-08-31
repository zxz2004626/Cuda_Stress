"""GPU integration smoke test: compile a real kernel, run it through cuda_runner,
verify the output blob (magic/canary/guards) and the CPU oracle match."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from generator.base import default_registry
from pipeline.config import Toolchain, load_config
from pipeline.executor import run_build, _close_persistent_runners

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "native" / "cuda_runner"
# Use the newest configured toolkit (supports the system gcc and C++17); older
# toolkits like cuda-10.1 need gcc-8 and C++14 and would fail these tests.
_cfg = load_config()
_tc = _cfg.toolchains[-1] if _cfg.toolchains else None
NVCC = shutil.which("nvcc") or (_tc.nvcc if _tc else "/usr/local/cuda/bin/nvcc")
ARCH = _tc.arch if _tc else "sm_75"
CCBIN = _tc.ccbin if _tc else "/usr/bin/gcc"


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
                       "boundary_words": 8}, "1.0.0", ARCH)
    src = tmp_path / "test.cu"
    src.write_text(case.source)
    (tmp_path / "input.bin").write_bytes(case.input_bytes)
    ptx = tmp_path / "k.ptx"
    r = subprocess.run([NVCC, f"-arch={ARCH}", "-ptx", "-o", str(ptx), str(src)],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    cubin = tmp_path / "k.cubin"
    r = subprocess.run([NVCC, f"-arch={ARCH}", "-cubin", "-Xptxas=-O3",
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
         "use_rare_ops": True, "use_immediates": True}, "1.0.0", ARCH)
    src = tmp_path / "test.cu"
    src.write_text(case.source)
    (tmp_path / "input.bin").write_bytes(case.input_bytes)
    cubin = tmp_path / "k.cubin"
    r = subprocess.run([NVCC, f"-arch={ARCH}", "-cubin", "-Xptxas=-O2",
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
    subprocess.run([NVCC, f"-arch={ARCH}", "-ptx", "-o", str(ptx), str(src)],
                   capture_output=True, timeout=180, check=True)
    cubin = tmp_path / "oob.cubin"
    subprocess.run([NVCC, f"-arch={ARCH}", "-cubin", "-Xptxas=-O3", "-o", str(cubin), str(src)],
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


def test_generator_ir_driver_checksum_matches_runtime_harness(tmp_path):
    """Dynamic ABI cubin path must reproduce generator_ir's Runtime checksum."""
    case = default_registry().get("generator_ir").generate(
        101, 202, {
            "template": "map_transform", "complexity": "low",
            "amplify_iters": 1, "codegen": False,
        }, "1.0.0", ARCH)
    test_dir = tmp_path / "case"
    build = test_dir / "builds" / "cuda-test__O3"
    build.mkdir(parents=True)
    src = test_dir / "test.cu"
    src.write_text(case.source)
    (test_dir / "input.bin").write_bytes(b"")
    (test_dir / "expected.bin").write_bytes(b"")
    (test_dir / "manifest.json").write_text(json.dumps(case.manifest()))

    root = str(Path(NVCC).resolve().parent.parent)
    ptxas = str(Path(root) / "bin" / "ptxas")
    ptx = build / "kernel.ptx"
    compiled = subprocess.run(
        [NVCC, f"-arch={ARCH}", "-std=c++17", "-ptx", "-o", str(ptx), str(src)],
        capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stderr
    assembled = subprocess.run(
        [ptxas, f"-arch={ARCH}", "-O3", "-o", str(build / "kernel.cubin"), str(ptx)],
        capture_output=True, text=True, timeout=180)
    assert assembled.returncode == 0, assembled.stderr

    cfg = load_config()
    cfg.execution["runner_restart_after"] = 4
    tc = Toolchain(
        id="cuda-test", root=root, nvcc=NVCC, ptxas=ptxas,
        nvdisasm=str(Path(root) / "bin" / "nvdisasm"),
        cuobjdump=str(Path(root) / "bin" / "cuobjdump"),
        arch=ARCH, ccbin=CCBIN, runnable=True)
    first = run_build(cfg, tc, case, build, 1)
    second = run_build(cfg, tc, case, build, 2)
    assert first.status == second.status == "OK"
    assert first.output_words == second.output_words
    assert first.execution["context_reused"] is True

    exe = tmp_path / "reference.exe"
    linked = subprocess.run(
        [NVCC, f"-arch={ARCH}", "-std=c++17", "-O2", "-Xptxas", "-O3",
         "-o", str(exe), str(src)], capture_output=True, text=True, timeout=180)
    assert linked.returncode == 0, linked.stderr
    reference = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert reference.returncode == 0, reference.stderr
    checksum = int(next(line.split("=", 1)[1] for line in reference.stdout.splitlines()
                        if line.startswith("checksum=")))
    assert first.output_words == struct.pack("<Q", checksum)
    _close_persistent_runners()
