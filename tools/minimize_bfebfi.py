#!/usr/bin/env python3
"""Minimize the over-MSB bfe/bfi case: truncate the program to find the
smallest prefix that still triggers the O0/O1 divergence."""

import copy, json, os, struct, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
NVCC = "/usr/local/cuda-12.6/bin/nvcc"
RUNNER = ROOT / "native" / "cuda_runner"
WORK = Path("/tmp/min_bfebfi"); WORK.mkdir(parents=True, exist_ok=True)

from pipeline.manifest import case_from_dir
from generator.strategies.inline_ptx import (
    _gen_program, _emit_cuda, _simulate, M32, M64
)
from generator.strategies.helpers import unpack_words

# load case
case_dir = ROOT / "corpus" / "differentials" / "inline_ptx__ebf45243624dc5d6"
m = json.loads((case_dir / "manifest.json").read_text())
seed = m["kernel_seed"]
full = _gen_program(seed, m["strategy_params"])
ib = (case_dir / "input.bin").read_bytes()
exp_original = (case_dir / "expected.bin").read_bytes()

def oracle_for(prog):
    ws = unpack_words(ib); exp = []
    for tid in range(64):
        w0=ws[(tid*3)%256]^seed; w1=ws[(tid*5+1)%256]^(seed>>11)
        w2=ws[(tid*7+2)%256]^(seed>>29); w3=ws[(tid*11+3)%256]^(seed>>37)
        a0=(w0^(w0>>32))&M32; a1=(w1^(w1>>32))&M32
        a2=(w2&M32)^((seed>>47)&M32); a3=w3&M32
        b0=(ws[(tid*13+4)%256]^(seed*3))&M64; b1=(ws[(tid*17+5)%256]^(seed>>13))&M64
        e0,e1 = _simulate(prog, a0,a1,a2,a3,b0,b1)
        exp.append(e0); exp.append(e1)
    return b''.join(struct.pack('<Q', x) for x in exp)

def run_gpu(prog, opt):
    (WORK/"t.cu").write_text(_emit_cuda(prog))
    (WORK/"in.bin").write_bytes(ib)
    p = subprocess.run([NVCC,"-arch=sm_86","-cubin",f"-Xptxas=-O{opt}",
        "-o",str(WORK/"k.cubin"),str(WORK/"t.cu")],
        capture_output=True, text=True, timeout=240)
    if p.returncode: return None
    r = subprocess.run([str(RUNNER),"--cubin",str(WORK/"k.cubin"),
        "--kernel","test_kernel","--input",str(WORK/"in.bin"),
        "--output",str(WORK/f"o{opt}.bin"),"--result",str(WORK/"e.json"),
        "--input-words","256","--output-words","128","--grid","1","--block","64",
        "--seed",str(seed),"--guards","8","--test-id","x","--timeout-ms","60000"],
        capture_output=True, text=True, timeout=120)
    if r.returncode: return None
    data = (WORK/f"o{opt}.bin").read_bytes()
    status = struct.unpack('<I', data[16:20])[0]
    return data[40:-8] if status == 0 else None

def diverges(prog):
    """O0==oracle AND O1!=oracle."""
    exp = oracle_for(prog)
    o0 = run_gpu(prog, 0)
    if o0 is None: return "CERR"
    if o0 != exp: return "O0_MISMATCH"
    o1 = run_gpu(prog, 1)
    if o1 is None: return "CERR"
    return "DIVERGE" if o1 != exp else "SAME"

# truncate: keep first K steps + output steps
def truncate(prog, K):
    c = copy.deepcopy(prog)
    out_steps = [s for s in c.steps if s["op"] == "out"]
    # keep first K non-out steps, then append the out steps
    kept = [s for s in c.steps if s["op"] != "out"][:K]
    c.steps = kept + out_steps
    return c

# binary search for the smallest K that diverges
out_idx = [i for i,s in enumerate(full.steps) if s["op"]=="out"]
total = len(full.steps) - len(out_idx)  # non-out steps
print(f"full program: {len(full.steps)} steps ({total} compute + {len(out_idx)} output)")
print(f"over-MSB bfi.b64 at step 168 (pos=13,width=71)\n")

# test a range of sizes to understand the threshold
sizes = [5, 10, 20, 40, 60, 80, 100, 120, 160, 200, 250, 300, 350, 400, total]
for K in sizes:
    t0 = time.monotonic()
    r = diverges(truncate(full, K))
    elapsed = time.monotonic() - t0
    print(f"prefix {K:3d} ({K*100//total:3d}%): {r:12s}  ({elapsed:.1f}s)")

# Fine binary search between the last "SAME" and first "DIVERGE"
print("\n--- fine binary search ---")
lo = 100  # last known SAME
hi = 160 # first known DIVERGE
while lo < hi:
    mid = (lo + hi) // 2
    r = diverges(truncate(full, mid))
    print(f"  prefix {mid:3d}: {r}")
    if r == "DIVERGE":
        hi = mid
    elif r == "SAME":
        lo = mid + 1
    else:
        print(f"  ERROR: {r}, retrying...")
        time.sleep(5)
        continue
print(f"\n=== minimal diverging prefix: {lo} steps ===")

# save minimal
minimal = truncate(full, lo)
src = _emit_cuda(minimal)
out = ROOT / "corpus" / "differentials" / "bug_reports" / "inline_ptx__ebf45243624dc5d6" / "minimal.cu"
Path(out).write_text(src)
print(f"minimal .cu written to: {out}")
print(f"minimal source size: {len(src)} chars, ~{len(src.splitlines())} lines")
