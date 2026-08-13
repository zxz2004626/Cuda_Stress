#!/usr/bin/env python3
"""
General inline-PTX bisection minimizer (works directly on test.cu, preserving %N
operand placeholders).

Usage:
    python3 tools/bisect_minimal.py <case_dir> [--nvcc /path/to/nvcc] [--no-write]

Steps:
  1. parse test.cu's asm volatile block into decl / setup / body / teardown
     (setup = leading mov.b32/mov.u64 block; teardown = mov.u64 %0/%1).
  2. verify the full body diverges -O0 vs -O1 on the GPU (nvcc + runner).
  3. bisect the body to the minimal subset that still diverges.
  4. write minimized_body.txt + minimization_report.json + minimal test.cu.
"""

import subprocess, tempfile, struct, json, hashlib, re, sys, shutil
from pathlib import Path

BLOB = struct.Struct("<QQIIQQ")
RUNNER = "/home/linux/cuda_stress/native/cuda_runner"

def parse_test_cu(text):
    """Extract the asm-volatile string literals, returning (decl, setup, body, teardown)
    as lists of raw instruction strings (with %% and %N placeholders)."""
    m = re.search(r'asm\s+volatile\s*\(', text)
    if not m:
        raise ValueError("no asm volatile block found")
    start = m.end()
    # the string section ends where the constraint list begins (': "=l"' / ': "r"' / ': "l"')
    cm = re.search(r':\s*"[=rl]', text[start:])
    end = start + cm.start() if cm else len(text)
    section = text[start:end]
    instrs = re.findall(r'"((?:[^"\\]|\\.)*)"', section)
    cleaned = []
    for ins in instrs:
        ins = re.sub(r'\\[nt]', '', ins)   # strip \n / \t escapes
        ins = ins.strip()
        if ins:
            cleaned.append(ins)

    decl, setup, body, teardown = [], [], [], []
    i = 0
    while i < len(cleaned) and cleaned[i].startswith('.reg'):
        decl.append(cleaned[i]); i += 1
    # leading mov.b32/mov.u64 block = setup (loads runtime values + constants)
    while i < len(cleaned) and re.match(r'mov\.(b32|u64)\s+%%x', cleaned[i]):
        setup.append(cleaned[i]); i += 1
    # teardown = trailing mov.u64 %0 / %1
    while i < len(cleaned) and (cleaned[-1].startswith('mov.u64 %0') or cleaned[-1].startswith('mov.u64 %1')):
        teardown.insert(0, cleaned.pop())
    body = cleaned[i:]
    return decl, setup, body, teardown

def generate_test_cu(case, decl, setup, body, teardown):
    """Reconstruct test.cu: original preamble + asm with given decl/setup/body/teardown,
    reusing the original output/input/clobber constraint lists verbatim."""
    orig = (case / "test.cu").read_text()
    pre = orig.split('asm volatile(')[0]
    # constraint tail: from ': "=l"(o0)' through the closing ');'
    ci = orig.find(': "=l"(o0)')
    if ci < 0:
        ci = orig.find(': "=l"')
    ei = orig.find(');', ci) + 2
    tail = orig[ci:ei].strip()
    post = orig[ei:]
    asm_lines = decl + setup + body + teardown
    asm_str = '\n        '.join(f'"{ins}"' for ins in asm_lines)
    return pre + 'asm volatile(\n        ' + asm_str + '\n        ' + tail + post

def compile_run(test_cu_text, opt, nvcc, seed, input_bin):
    key = (test_cu_text, opt, nvcc)
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        src = tdir/"t.cu"; cubin = tdir/"k.cubin"; out = tdir/"out.bin"
        src.write_text(test_cu_text)
        r = subprocess.run([nvcc, "-arch=sm_86", "-cubin", f"-Xptxas=-O{opt}", "-o", str(cubin), str(src)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"nvcc:{r.stderr[-120:]}"
        r = subprocess.run([RUNNER, "--cubin", str(cubin), "--kernel", "test_kernel",
            "--input", str(input_bin), "--output", str(out), "--result", str(tdir/"r.json"),
            "--input-words", "256", "--output-words", "128", "--grid", "1", "--block", "64",
            "--seed", str(seed), "--guards", "8", "--test-id", "bisect", "--timeout-ms", "60000"],
            capture_output=True, text=True, timeout=70)
        if r.returncode != 0 or not out.exists():
            return None, f"runner:{r.stderr[-120:]}"
        return hashlib.sha256(out.read_bytes()[BLOB.size:-8]).hexdigest(), None

def diverges(case, decl, setup, body, teardown, nvcc, seed, input_bin):
    cu = generate_test_cu(case, decl, setup, body, teardown)
    o0, e0 = compile_run(cu, 0, nvcc, seed, input_bin)
    o1, e1 = compile_run(cu, 1, nvcc, seed, input_bin)
    if o0 is None or o1 is None:
        return False, f"{e0 or ''} {e1 or ''}"
    return (o0 != o1), f"O0={o0[:8]} O1={o1[:8]}"

def bisect(case, decl, setup, body, teardown, nvcc, seed, input_bin):
    current = list(body)
    step = 0
    while len(current) > 2:
        step += 1
        half = len(current)//2
        for cand in (current[half:], current[:half]):
            d, info = diverges(case, decl, setup, cand, teardown, nvcc, seed, input_bin)
            if d:
                current = cand
                print(f"  [{step}] -> {len(current)}  {info}")
                break
        else:
            removed = False
            for i in range(len(current)-1, -1, -1):
                d, info = diverges(case, decl, setup, current[:i]+current[i+1:], teardown, nvcc, seed, input_bin)
                if d:
                    current = current[:i]+current[i+1:]
                    print(f"  [{step}] -1 -> {len(current)}  {info}")
                    removed = True
                    break
            if not removed:
                print(f"  [{step}] minimal at {len(current)}")
                break
    return current

def main():
    case = Path(sys.argv[1]).resolve()
    nvcc = "/usr/local/cuda-11.8/bin/nvcc"
    write = True
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--nvcc':
            nvcc = args[i+1]; i += 2
        elif args[i] == '--no-write':
            write = False; i += 1
        else:
            i += 1

    manifest = json.loads((case/"manifest.json").read_text())
    seed = manifest["kernel_seed"]
    input_bin = case/"input.bin"

    src_text = (case/"test.cu").read_text()
    decl, setup, body, teardown = parse_test_cu(src_text)
    print(f"Case: {case.name}")
    print(f"  decl={len(decl)} setup={len(setup)} body={len(body)} teardown={len(teardown)}")

    d, info = diverges(case, decl, setup, body, teardown, nvcc, seed, input_bin)
    print(f"  full body diverges: {d}  {info}")
    if not d:
        print("  ERROR: full body does not diverge with this nvcc")
        sys.exit(1)

    print("Bisecting...")
    mini = bisect(case, decl, setup, body, teardown, nvcc, seed, input_bin)
    print(f"\n=== MINIMIZED: {len(body)} -> {len(mini)} ===")
    for ins in mini:
        print(f"  {ins}")

    (case/"minimized_body.txt").write_text('\n'.join(mini) + '\n')
    ops = set()
    for x in mini:
        mm = re.match(r'^(\w+)\.', x)
        if mm: ops.add(mm.group(1))
    (case/"minimization_report.json").write_text(json.dumps({
        "original_count": len(body), "minimized_count": len(mini),
        "instructions": mini, "opcodes": sorted(ops),
    }, indent=2))
    print(f"\nWrote minimized_body.txt + minimization_report.json")

    if write:
        if not (case/"test.cu.orig").exists():
            shutil.copy2(case/"test.cu", case/"test.cu.orig")
        cu = generate_test_cu(case, decl, setup, mini, teardown)
        (case/"test.cu").write_text(cu)
        print(f"Wrote minimal test.cu ({len(cu)} bytes); original -> test.cu.orig")

if __name__ == "__main__":
    main()
