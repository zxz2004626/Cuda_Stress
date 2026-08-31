#!/usr/bin/env python3
"""
Bisection minimizer for inline_ptx__b4a84ffec04cd1a5.

Multi-family combinatorial ptxas O1+ wrong-code bug. Reproduces the divergence
by rebuilding PTX from subsets of the inline assembly, compiling O0 vs O1, running
on GPU, and comparing output hashes.

Key reproduction fact (verified): ALL toolchains 11.3..12.6 produce the SAME
wrong output at O1+ (hash a270950752e7) vs correct at O0 (hash 933ac78a969b).
The README's claim that 11.3 is correct is WRONG — the bug is cross-toolchain.
"""

import subprocess, tempfile, struct, json, hashlib, re, sys
from pathlib import Path

CASE_DIR = Path("/home/linux/cuda_stress/corpus/differentials/bug_reports/inline_ptx__b4a84ffec04cd1a5")
PTXAS = "/usr/local/cuda-11.8/bin/ptxas"
RUNNER = "/home/linux/cuda_stress/native/cuda_runner"
INPUT_BIN = CASE_DIR / "input.bin"
MANIFEST = json.loads((CASE_DIR / "manifest.json").read_text())

BLOB_HEADER = struct.Struct("<QQIIQQ")
_cache = {}

def compile_and_run(ptx_text, opt, ptxas=PTXAS):
    key = (ptx_text, opt, ptxas)
    if key in _cache:
        return _cache[key]
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        ptx_path = tdir / "k.ptx"
        cubin_path = tdir / "k.cubin"
        out_path = tdir / "out.bin"
        ptx_path.write_text(ptx_text)
        r = subprocess.run([ptxas, "-arch=sm_86", f"-O{opt}", "-o", str(cubin_path), str(ptx_path)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            _cache[key] = (None, f"ptxas:{r.stderr[:100]}")
            return _cache[key]
        r = subprocess.run([
            RUNNER, "--cubin", str(cubin_path), "--kernel", "test_kernel",
            "--input", str(INPUT_BIN), "--output", str(out_path), "--result", str(tdir/"r.json"),
            "--input-words", str(MANIFEST['input_word_count']), "--output-words", str(MANIFEST['output_word_count']),
            "--grid", "1", "--block", "64", "--seed", str(MANIFEST['kernel_seed']),
            "--guards", "8", "--test-id", "x", "--timeout-ms", "60000",
        ], capture_output=True, text=True, timeout=70)
        if r.returncode != 0 or not out_path.exists():
            _cache[key] = (None, f"runner:{r.stderr[:100]}")
            return _cache[key]
        data = out_path.read_bytes()
        h = hashlib.sha256(data[BLOB_HEADER.size:-8]).hexdigest()
        _cache[key] = (h, None)
        return _cache[key]

def check_divergence(body):
    ptx = rebuild(body)
    o0, e0 = compile_and_run(ptx, 0)
    o1, e1 = compile_and_run(ptx, 1)
    if o0 is None or o1 is None:
        return False, f"{e0 or ''} {e1 or ''}"
    return (o0 != o1), f"O0={o0[:8]} O1={o1[:8]}"

# Global parsed components
PREAMBLE = []
DECLS = []
SETUP = []
TEARDOWN = []
POSTAMBLE = []

def parse_ptx(ptx_text):
    global PREAMBLE, DECLS, SETUP, TEARDOWN, POSTAMBLE
    lines = ptx_text.splitlines()
    state = "pre"
    body = []
    for line in lines:
        s = line.rstrip('\n')
        if 'begin inline asm' in s:
            PREAMBLE.append(s)
            state = "decls"
            continue
        if 'end inline asm' in s:
            state = "post"
            POSTAMBLE.append(s)
            continue
        if state == "pre":
            PREAMBLE.append(s)
        elif state == "post":
            POSTAMBLE.append(s)
        else:
            t = s.strip()
            if not t:
                continue
            if t.startswith('.reg '):
                DECLS.append(t)
            elif (t.startswith('mov.b32 %xr') or t.startswith('mov.u64 %xd')):
                SETUP.append(t)
            elif (t.startswith('mov.u64 %rd') and 'inline' not in t):
                TEARDOWN.append(t)
            else:
                body.append(t)
    return body

def rebuild(body):
    parts = list(PREAMBLE)
    parts += DECLS + SETUP + body + TEARDOWN
    parts += POSTAMBLE
    return '\n'.join(parts) + '\n'

def main():
    ptx_text = (CASE_DIR / "test.cu").read_text()
    # nvcc generates PTX; get it fresh
    import subprocess as sp
    r = sp.run(["/usr/local/cuda-11.8/bin/nvcc", "-arch=sm_86", "-ptx", "-o", "/tmp/b4a84.ptx",
                str(CASE_DIR / "test.cu")], capture_output=True, text=True)
    ptx_text = Path("/tmp/b4a84.ptx").read_text()

    body = parse_ptx(ptx_text)
    print(f"Parsed {len(body)} body instructions")
    print(f"  decls={len(DECLS)} setup={len(SETUP)} teardown={len(TEARDOWN)}")

    # Verify full set reproduces
    div, info = check_divergence(body)
    print(f"Full set ({len(body)}): diverges={div}  {info}")
    if not div:
        print("ERROR: full set doesn't reproduce")
        sys.exit(1)

    # Save full body for reference
    Path(CASE_DIR / "full_body.txt").write_text('\n'.join(body))

    # Bisect
    current = list(body)
    step = 0
    while len(current) > 2:
        step += 1
        half = len(current) // 2
        # Try second half
        cand = current[half:]
        div, info = check_divergence(cand)
        if div:
            current = cand
            print(f"  [{step}] keep 2nd half ({len(current)}) {info}")
            continue
        cand = current[:half]
        div, info = check_divergence(cand)
        if div:
            current = cand
            print(f"  [{step}] keep 1st half ({len(current)}) {info}")
            continue
        # Neither half alone — try removing one instruction at a time
        removed = False
        for i in range(len(current)-1, -1, -1):
            test = current[:i] + current[i+1:]
            div, info = check_divergence(test)
            if div:
                current = test
                print(f"  [{step}] removed single instr {i} ({len(current)}) {info}")
                removed = True
                break
        if not removed:
            print(f"  [{step}] no single removal possible, minimal at {len(current)}")
            break

    print(f"\n=== Minimized: {len(body)} -> {len(current)} instructions ===")
    for i, instr in enumerate(current):
        print(f"  [{i}] {instr}")

    out = CASE_DIR / "minimized_body.txt"
    out.write_text('\n'.join(current))
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
