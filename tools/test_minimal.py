#!/usr/bin/env python3
"""
Test a candidate minimal inline-asm body for a bug-report case: generate a full
test.cu, compile -O0 / -O1 on several toolchains, run on GPU, and report divergence.
Also dump the O0 vs O1 SASS for the given opcode so the root cause can be eyeballed.

Usage: python3 tools/test_minimal.py <case_dir> <body.txt>
  body.txt = one PTX instruction per line (%%xr / %N placeholders), WITHOUT the
             .reg decls or output movs (those are added automatically).
"""
import subprocess, tempfile, struct, hashlib, json, sys, re
from pathlib import Path

RUNNER = "/home/linux/cuda_stress/native/cuda_runner"
BLOB = struct.Struct("<QQIIQQ")

def preamble_and_constraints(case):
    orig = (case / "test.cu").read_text()
    pre = orig.split('asm volatile(')[0]
    ci = orig.find(': "=l"(o0)')
    ei = orig.find(');', ci) + 2
    return pre, orig[ci:ei].strip(), orig[ei:]

def gen_test_cu(case, body_lines):
    pre, tail, post = preamble_and_constraints(case)
    decl = ['.reg .b32 %%xr<12>;', '.reg .b64 %%xd<6>;', '.reg .pred %%xp<5>;']
    # ensure output movs present
    if not any('mov.u64 %0' in l for l in body_lines):
        body_lines = body_lines + ['mov.u64 %0, %%xd0;', 'mov.u64 %1, %%xd1;']
    all_lines = decl + body_lines
    asm_str = '\n        '.join(f'"{l}"' for l in all_lines)
    return pre + 'asm volatile(\n        ' + asm_str + '\n        ' + tail + post

def compile_run(cu, opt, nvcc, seed, input_bin):
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp); src = t/"t.cu"; c = t/"k.cubin"; o = t/"out.bin"
        src.write_text(cu)
        r = subprocess.run([nvcc, "-arch=sm_86", "-cubin", f"-Xptxas=-O{opt}", "-o", str(c), str(src)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"nvcc:{r.stderr[-160:]}"
        r = subprocess.run([RUNNER, "--cubin", str(c), "--kernel", "test_kernel",
            "--input", str(input_bin), "--output", str(o), "--result", str(t/"r.json"),
            "--input-words", "256", "--output-words", "128", "--grid", "1", "--block", "64",
            "--seed", str(seed), "--guards", "8", "--test-id", "x", "--timeout-ms", "60000"],
            capture_output=True, text=True, timeout=70)
        if r.returncode != 0 or not o.exists():
            return None, f"runner:{r.stderr[-160:]}"
        return hashlib.sha256(o.read_bytes()[BLOB.size:-8]).hexdigest()[:12], None

def main():
    case = Path(sys.argv[1]).resolve()
    body_file = Path(sys.argv[2])
    body_lines = [l.strip() for l in body_file.read_text().splitlines() if l.strip()]
    manifest = json.loads((case/"manifest.json").read_text())
    seed = manifest["kernel_seed"]
    input_bin = case/"input.bin"

    cu = gen_test_cu(case, body_lines)
    print(f"body instructions: {len(body_lines)}")
    ok = True
    for lbl, nvcc in [("11.3","/usr/local/cuda-11.3/bin/nvcc"),("11.8","/usr/local/cuda-11.8/bin/nvcc"),("12.6","/usr/local/cuda-12.6/bin/nvcc")]:
        h0,e0 = compile_run(cu, 0, nvcc, seed, input_bin)
        h1,e1 = compile_run(cu, 1, nvcc, seed, input_bin)
        trig = h0 and h1 and h0 != h1
        if not trig: ok = False
        print(f"  {lbl}: O0={h0} O1={h1} {'✓' if trig else '✗ ' + str(e0 or e1)}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
