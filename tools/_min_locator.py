#!/usr/bin/env python3
"""Locate which variable in a generator_ir kernel's _fv XOR chain diverges
between ptxas -O0 and -O1.  Outputs each candidate variable alone and compares
the standalone harness checksum at O0 vs O1."""
import re, subprocess, sys, tempfile, hashlib
from pathlib import Path

NVCC = "/usr/local/cuda-12.6/bin/nvcc"
SRC = Path(sys.argv[1])

text = SRC.read_text()

# candidates: variables XORed into _fv (lines `_fv = (_fv ^ ((unsigned int)(X)));`)
xor = re.findall(r'_fv = \(_fv \^ \(\(unsigned int\)\((\w+)\)\)\);', text)
cands = list(dict.fromkeys(xor))   # dedupe, keep order

def build(src: str, opt: int, tag: str) -> str:
    d = Path(tempfile.mkdtemp(prefix="min_"))
    cu = d / "t.cu"; cu.write_text(src)
    exe = d / f"o{opt}"
    r = subprocess.run([NVCC, "-arch=sm_86", f"-Xptxas=-O{opt}", "-o", str(exe), str(cu)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        return None, r.stderr[-800:]
    return str(exe), None

def run_checksum(exe: str) -> str:
    r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    m = re.search(r'checksum=(\d+)', r.stdout)
    return m.group(1) if m else f"ERR:{r.returncode}:{r.stderr[-120:]}"

def diverges(src: str) -> tuple[bool, str]:
    e0, err0 = build(src, 0, "0")
    if e0 is None: return False, f"compileO0:{err0[:200]}"
    e1, err1 = build(src, 1, "1")
    if e1 is None: return False, f"compileO1:{err1[:200]}"
    c0 = run_checksum(e0); c1 = run_checksum(e1)
    if not c0.isdigit() or not c1.isdigit():
        return False, f"run:{c0}|{c1}"
    return c0 != c1, f"O0={c0} O1={c1}"

# 1. baseline: full kernel must diverge
ok, detail = diverges(text)
print(f"baseline (full kernel): diverge={ok}  {detail}")
if not ok:
    print("baseline does not diverge — cannot minimize"); sys.exit(1)

# 2. test each candidate variable alone
print(f"\ntesting {len(cands)} candidates:")
for v in cands:
    # replace the final output write with just this variable
    mod = re.sub(r'output\[tid\] = _fv;', f'output[tid] = ((unsigned int)({v}));', text, count=1)
    if f'output[tid] = ((unsigned int)({v}));' not in mod:
        print(f"  {v:10s} SKIP (write not replaced)"); continue
    d, det = diverges(mod)
    mark = "DIVERGES" if d else "same"
    print(f"  {v:10s} {mark:9s} {det}")
