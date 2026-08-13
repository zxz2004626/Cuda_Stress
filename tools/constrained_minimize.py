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
    def is_setup(ins):
        if re.match(r'mov\.(b32|u64)\s+%%x', ins):
            return True
        m = re.match(r'xor\.b32\s+(%%xr\d+),\s*(%%xr\d+),\s*(%%xr\d+)', ins)
        if m and m.group(1) == m.group(2) == m.group(3):
            return True  # self-xor = zero (register init)
        return False

    decl, setup, body, teardown = [], [], [], []
    i = 0
    while i < len(cleaned) and cleaned[i].startswith('.reg'):
        decl.append(cleaned[i]); i += 1
    while i < len(cleaned) and is_setup(cleaned[i]):
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
        for attempt in range(3):
            try:
                r = subprocess.run([RUNNER, "--cubin", str(c), "--kernel", "test_kernel",
                    "--input", str(input_bin), "--output", str(o), "--result", str(t/"r.json"),
                    "--input-words", "256", "--output-words", "128", "--grid", "1", "--block", "64",
                    "--seed", str(seed), "--guards", "8", "--test-id", "x", "--timeout-ms", "30000"], capture_output=True, text=True, timeout=120)
                break
            except subprocess.TimeoutExpired:
                return None, "runner-timeout"
            except FileNotFoundError:
                if attempt == 2:
                    return None, "runner-missing"
                import time as _t; _t.sleep(2)
        else:
            return None, "runner-missing"
        if r.returncode != 0 or not o.exists(): return None, r.stderr[-120:]
        return hashlib.sha256(o.read_bytes()[BLOB.size:-8]).hexdigest()[:12], None

def find_uninit(setup, body):
    """Return registers read (as source) before being written (as dest), given `setup`.
    A non-empty result means the body has undefined-behaviour (read-before-write)."""
    initialized = set()
    for ins in setup:
        regs = re.findall(r'%x[rp][0-9]+', ins)
        if regs:
            initialized.add(regs[0])
    uninit = []
    for ins in body:
        regs = re.findall(r'%x[rp][0-9]+', ins)
        if not regs:
            continue
        for src in regs[1:]:
            if src not in initialized and src not in uninit:
                uninit.append(src)
        initialized.add(regs[0])
    return uninit

def sanitize_setup(setup):
    """Replace self-xor-zero (xor.b32 %xN,%xN,%xN — reads uninitialised register) with
    an equivalent UB-free mov.b32 %xN, 0."""
    out = []
    for ins in setup:
        m = re.match(r'xor\.b32\s+(%%xr\d+),\s*(%%xr\d+),\s*(%%xr\d+)', ins)
        if m and m.group(1) == m.group(2) == m.group(3):
            out.append(f'mov.b32 {m.group(1)}, 0;')
        else:
            out.append(ins)
    return out

def diverges(orig, decl, setup, body, teardown, nvcc, seed, input_bin):
    uninit = find_uninit(setup, body)
    if uninit:
        return False, f"uninit:{','.join(uninit)}"
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
    setup = sanitize_setup(setup)  # remove self-xor-zero UB in the setup

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

    # Generic delta-debug: minimize `cands` while the predicate `pred(subset)` holds,
    # keeping `fixed` instructions always present. Handles half==0 without looping.
    def ddmin(cands, fixed, label, pred):
        current = list(cands)
        step = 0
        while current:
            step += 1
            progressed = False
            half = len(current) // 2
            for cand in (current[half:], current[:half]):
                if not cand or len(cand) == len(current):
                    continue  # skip trivial/no-op candidates (avoids infinite loop)
                d2, i2 = pred(fixed + cand)
                if d2:
                    current = cand
                    print(f"  [{label} {step}] -> {len(current)}  {i2}")
                    progressed = True
                    break
            if not progressed:
                for i in range(len(current) - 1, -1, -1):
                    cand = current[:i] + current[i+1:]
                    d2, i2 = pred(fixed + cand)
                    if d2:
                        current = cand
                        print(f"  [{label} {step}] -1 -> {len(current)}  {i2}")
                        progressed = True
                        break
            if not progressed:
                print(f"  [{label} {step}] minimal at {len(current)}")
                break
        return current

    # Phase 2: minimize the scaffold (pin all root instructions)
    root_instrs = [i for i in kept if root in i]
    scaffold = [i for i in kept if root not in i]
    print(f"  root instrs={len(root_instrs)}, scaffold={len(scaffold)}; bisecting scaffold...")
    pred_scaffold = lambda cand: diverges(orig, decl, setup, cand, teardown, nvcc, seed, input_bin)
    min_scaffold = ddmin(scaffold, root_instrs, "scaffold", pred_scaffold)

    # Phase 3: minimize the root instructions (pin the minimal scaffold)
    print(f"  minimizing root instructions (of {len(root_instrs)})...")
    pred_root = lambda cand: diverges(orig, decl, setup, cand + min_scaffold, teardown, nvcc, seed, input_bin)
    min_root = ddmin(root_instrs, min_scaffold, "root", pred_root)

    mini = min_root + min_scaffold
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
