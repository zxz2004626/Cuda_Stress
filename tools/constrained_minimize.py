#!/usr/bin/env python3
"""
Constrained minimizer that PRESERVES a case's documented root cause.

Key insight: every inline_ptx body mixes several rare opcodes, and `bfi.b32(7,39)`
(over-MSB) diverges on its own.  A naive DELETE-based bisection removes the root
cause and keeps the incidental bfi, collapsing every case onto the bfi bug.

Correct approach:
  1. REPLACE every non-root, non-scaffold opcode with a `mov` identity (this kills
     the incidental bfi/bfe/... bugs WITHOUT breaking register liveness),
  2. verify the case still diverges O0 vs O1 (i.e. the root cause is real),
  3. then bisect (DELETE) the remaining scaffold, never removing the root cause.

Usage: python3 tools/constrained_minimize.py <case_dir> <root_substr>
"""

import subprocess, tempfile, struct, json, hashlib, re, sys, shutil
from pathlib import Path

RUNNER = "/home/linux/cuda_stress/native/cuda_runner"
BLOB = struct.Struct("<QQIIQQ")

SCAFFOLD = {'mov', 'cvt', 'xor', 'add', 'sub'}
# ops with their OWN divergence bugs — neutralised so they can't hijack the bisection.
# (comparison/select/arithmetic: set/setp/selp/mul/div/rem/min/max/not/and/or/shl/shr
#  are KEPT as potential scaffold for the root cause.)
NEUTRALISE = {'bfi','bfe','bfind','brev','clz','cnot','lop3','popc','sad','slct',
              'shf','prmt','mul24','mad24','abs','neg'}

def parse_test_cu(text):
    m = re.search(r'asm\s+volatile\s*\(', text)
    start = m.end()
    cm = re.search(r':\s*"[=rl]', text[start:])
    end = start + cm.start() if cm else len(text)
    section = text[start:end]
    instrs = re.findall(r'"((?:[^"\\]|\\.)*)"', section)
    cleaned = [re.sub(r'\\[nt]', '', i).strip() for i in instrs if re.sub(r'\\[nt]', '', i).strip()]
    decl, setup, body, teardown = [], [], [], []
    i = 0
    while i < len(cleaned) and cleaned[i].startswith('.reg'):
        decl.append(cleaned[i]); i += 1
    while i < len(cleaned) and re.match(r'mov\.(b32|u64)\s+%%x', cleaned[i]):
        setup.append(cleaned[i]); i += 1
    while i < len(cleaned) and (cleaned[-1].startswith('mov.u64 %0') or cleaned[-1].startswith('mov.u64 %1')):
        teardown.insert(0, cleaned.pop())
    body = cleaned[i:]
    return decl, setup, body, teardown

def opcode(ins):
    m = re.match(r'^([a-z0-9]+)\.', ins); return m.group(1) if m else ''

def identity(ins):
    """Replace an instruction with a self-mov identity (dest = dest), which is
    always valid regardless of operand widths. Preserves register liveness."""
    m = re.match(r'(\w+(?:\.\w+)+)\s+(%%[a-z0-9]+)', ins)
    if m:
        dest = m.group(2)
        dtype = 'b32' if dest.startswith('%%xr') else 'b64'
        return f'mov.{dtype} {dest}, {dest};'
    return ins

def gen_test_cu(orig, decl, setup, body, teardown):
    pre = orig.split('asm volatile(')[0]
    ci = orig.find(': "=l"(o0)'); ei = orig.find(');', ci) + 2
    tail, post = orig[ci:ei].strip(), orig[ei:]
    asm_str = '\n        '.join(f'"{l}"' for l in decl + setup + body + teardown)
    return pre + 'asm volatile(\n        ' + asm_str + '\n        ' + tail + post

def compile_run(cu, opt, nvcc, seed, input_bin):
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp); src = t/"t.cu"; c = t/"k.cubin"; o = t/"out.bin"
        src.write_text(cu)
        try:
            r = subprocess.run([nvcc, "-arch=sm_86", "-cubin", f"-Xptxas=-O{opt}", "-o", str(c), str(src)], capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None, "nvcc-timeout"
        if r.returncode != 0: return None, r.stderr[-120:]
        try:
            r = subprocess.run([RUNNER, "--cubin", str(c), "--kernel", "test_kernel",
                "--input", str(input_bin), "--output", str(o), "--result", str(t/"r.json"),
                "--input-words", "256", "--output-words", "128", "--grid", "1", "--block", "64",
                "--seed", str(seed), "--guards", "8", "--test-id", "x", "--timeout-ms", "30000"], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None, "runner-timeout"
        if r.returncode != 0 or not o.exists(): return None, r.stderr[-120:]
        return hashlib.sha256(o.read_bytes()[BLOB.size:-8]).hexdigest()[:12], None

def diverges(orig, decl, setup, body, teardown, nvcc, seed, input_bin):
    cu = gen_test_cu(orig, decl, setup, body, teardown)
    o0, e0 = compile_run(cu, 0, nvcc, seed, input_bin)
    o1, e1 = compile_run(cu, 1, nvcc, seed, input_bin)
    if o0 is None or o1 is None: return False, f"{e0 or ''} {e1 or ''}"
    return o0 != o1, f"O0={o0} O1={o1}"

def main():
    case = Path(sys.argv[1]).resolve(); root = sys.argv[2]
    nvcc = "/usr/local/cuda-11.8/bin/nvcc"
    manifest = json.loads((case/"manifest.json").read_text())
    seed = manifest["kernel_seed"]; input_bin = case/"input.bin"
    orig = (case/"test.cu").read_text()
    decl, setup, body, teardown = parse_test_cu(orig)

    # Phase 1: neutralise every non-root, non-scaffold, bit-manip op (replace with mov identity)
    # so its incidental divergence bug is removed WITHOUT breaking register liveness.
    from collections import Counter
    kept = []
    neutralised_ops = Counter()
    for ins in body:
        op = opcode(ins)
        if root in ins or op in SCAFFOLD:
            kept.append(ins)
        elif op in NEUTRALISE:
            kept.append(identity(ins))
            neutralised_ops[op] += 1
        else:
            kept.append(ins)  # set/setp/selp/mul/div/rem/... kept as potential scaffold
    print(f"{case.name}: body={len(body)} -> kept={len(kept)} (neutralised {sum(neutralised_ops.values())} bit-manip ops: {dict(neutralised_ops)})")

    d, info = diverges(orig, decl, setup, kept, teardown, nvcc, seed, input_bin)
    print(f"  after neutralising bit-manip ops, diverges: {d}  {info}")
    if not d:
        print("  -> root cause is genuinely combinatorial (needs other ops); cannot isolate further.")
        sys.exit(2)

    # Phase 2: bisect the scaffold (DELETE), pinning root cause
    root_instrs = [i for i in kept if root in i]
    scaffold = [i for i in kept if root not in i]
    print(f"  root instrs={len(root_instrs)}, scaffold={len(scaffold)}; bisecting...")
    current = list(scaffold)
    step = 0
    while current:
        step += 1
        half = len(current)//2
        progressed = False
        for cand in (current[half:], current[:half]):
            d2, i2 = diverges(orig, decl, setup, root_instrs + cand, teardown, nvcc, seed, input_bin)
            if d2:
                current = cand; print(f"  [{step}] -> {len(current)} scaffold  {i2}"); progressed = True; break
        if not progressed:
            for i in range(len(current)-1, -1, -1):
                d2, i2 = diverges(orig, decl, setup, root_instrs + current[:i] + current[i+1:], teardown, nvcc, seed, input_bin)
                if d2:
                    current = current[:i] + current[i+1:]; print(f"  [{step}] -1 -> {len(current)}  {i2}"); progressed = True; break
        if not progressed:
            print(f"  [{step}] minimal scaffold at {len(current)}"); break

    mini = root_instrs + current
    print(f"\n=== MINIMIZED (root cause '{root}' preserved): {len(body)} -> {len(mini)} ===")
    for ins in mini: print(f"  {ins}")

    (case/"minimized_body.txt").write_text('\n'.join(mini) + '\n')
    (case/"minimization_report.json").write_text(json.dumps({
        "original_count": len(body), "minimized_count": len(mini),
        "root_cause": root, "instructions": mini,
        "opcodes": sorted({opcode(x) for x in mini if opcode(x)}),
    }, indent=2))
    if not (case/"test.cu.orig").exists():
        shutil.copy2(case/"test.cu", case/"test.cu.orig")
    (case/"test.cu").write_text(gen_test_cu(orig, decl, setup, mini, teardown))
    print(f"Wrote minimal test.cu; original -> test.cu.orig")

if __name__ == "__main__":
    main()
