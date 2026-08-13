#!/usr/bin/env python3
"""
Regenerate expected.bin (CPU-oracle equivalent) for a MINIMIZED test.cu.

The minimized test.cu reproduces the O0-vs-O1 divergence but its O0 output no longer
matches the ORIGINAL expected.bin (which belongs to the pre-minimization program).
We regenerate the oracle from the O0 output on CUDA 11.3 (correct for all cases),
and update manifest.json's source/expected hashes so `compare` works again.

Usage: python3 tools/regenerate_oracle.py <case_dir>
"""
import subprocess, tempfile, struct, json, hashlib, sys
from pathlib import Path

RUNNER = "/home/linux/cuda_stress/native/cuda_runner"
BLOB = struct.Struct("<QQIIQQ")  # magic, test_id_hash, status, reserved, words, canary

def main():
    case = Path(sys.argv[1]).resolve()
    nvcc = "/usr/local/cuda-11.3/bin/nvcc"  # 11.3 O0 is correct for every known case
    manifest_path = case / "manifest.json"
    m = json.loads(manifest_path.read_text())
    seed = m["kernel_seed"]
    input_bin = case / "input.bin"
    test_cu = case / "test.cu"

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        cubin = t / "k.cubin"
        out = t / "out.bin"
        r = subprocess.run([nvcc, "-arch=sm_86", "-cubin", "-Xptxas=-O0",
                            "-o", str(cubin), str(test_cu)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"COMPILE ERROR: {r.stderr[-200:]}"); sys.exit(1)
        r = subprocess.run([RUNNER, "--cubin", str(cubin), "--kernel", "test_kernel",
            "--input", str(input_bin), "--output", str(out), "--result", str(t/"r.json"),
            "--input-words", str(m["input_word_count"]), "--output-words", str(m["output_word_count"]),
            "--grid", "1", "--block", "64", "--seed", str(seed),
            "--guards", "8", "--test-id", "regenerate", "--timeout-ms", "30000"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not out.exists():
            print(f"RUN ERROR: {r.stderr[-200:]}"); sys.exit(1)
        blob = out.read_bytes()
        body = blob[BLOB.size:-8]  # strip 40-byte header and 8-byte tail canary

    expected_bin = case / "expected.bin"
    expected_bin.write_bytes(body)
    print(f"regenerated expected.bin ({len(body)} bytes) from 11.3 -O0")

    # update manifest hashes (source + expected)
    src_sha = hashlib.sha256(test_cu.read_bytes()).hexdigest()
    exp_sha = hashlib.sha256(body).hexdigest()
    m["hashes"]["source"] = f"sha256:{src_sha}"
    m["hashes"]["expected"] = f"sha256:{exp_sha}"
    # the minimized program is not a generator output; drop the stale oracle hook
    m["cpu_oracle"] = None
    m["case_id"] = f"inline_ptx__{src_sha[:16]}"
    m["strategy"] = "inline_ptx_minimized"
    m["strategy_params"] = {}
    manifest_path.write_text(json.dumps(m, indent=2) + "\n")
    print(f"updated manifest.json: source={src_sha[:16]} expected={exp_sha[:16]} case_id={m['case_id']}")

if __name__ == "__main__":
    main()
