#!/usr/bin/env python3
"""
Bisection minimizer for inline_ptx__4e889aef7c32d967.

Reproduces the CUDA 11.3 ptxas -O1 bug by:
1. Rebuilding PTX from subsets of the inline assembly
2. Compiling with ptxas 11.3 at O0 and O1
3. Running both on GPU
4. Comparing output hashes
5. Binary-searching to find the minimum instruction set that triggers the bug
"""

import subprocess, tempfile, struct, json, hashlib, re, sys, shutil
from pathlib import Path

CASE_DIR = Path("/home/linux/cuda_stress/runs/run-20260812-115625-1149/differentials/inline_ptx__4e889aef7c32d967")
PTXAS_113 = "/usr/local/cuda-11.3/bin/ptxas"
PTXAS_118 = "/usr/local/cuda-11.8/bin/ptxas"
RUNNER = "/home/linux/cuda_stress/native/cuda_runner"
INPUT_BIN = CASE_DIR / "input.bin"
MANIFEST = json.loads((CASE_DIR / "manifest.json").read_text())

BLOB_HEADER = struct.Struct("<QQIIQQ")

# Cache for O0 results (O0 is deterministic given PTX)
_o0_cache = {}

def compile_and_run(ptx_text: str, opt: int, ptxas: str = PTXAS_113) -> tuple[str | None, str | None]:
    """Returns (output_hash, error)."""
    cache_key = (ptx_text, opt, ptxas)
    if cache_key in _o0_cache:
        return _o0_cache[cache_key]

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        ptx_path = tdir / "kernel.ptx"
        cubin_path = tdir / "kernel.cubin"
        out_path = tdir / "output.bin"
        result_path = tdir / "result.json"
        ptx_path.write_text(ptx_text)

        r = subprocess.run([ptxas, "-arch=sm_86", f"-O{opt}",
                           "-o", str(cubin_path), str(ptx_path)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            result = (None, f"ptxas: {r.stderr[:200]}")
            _o0_cache[cache_key] = result
            return result

        r = subprocess.run([
            RUNNER,
            "--cubin", str(cubin_path),
            "--kernel", "test_kernel",
            "--input", str(INPUT_BIN),
            "--output", str(out_path),
            "--result", str(result_path),
            "--input-words", str(MANIFEST['input_word_count']),
            "--output-words", str(MANIFEST['output_word_count']),
            "--grid", "1",
            "--block", "64",
            "--seed", str(MANIFEST['kernel_seed']),
            "--guards", "8",
            "--test-id", MANIFEST['case_id'],
            "--device", "0",
            "--timeout-ms", "30000",
        ], capture_output=True, text=True, timeout=35)

        if r.returncode != 0 or not out_path.exists():
            result = (None, f"runner: {r.stderr[:300]}")
            _o0_cache[cache_key] = result
            return result

        out_data = out_path.read_bytes()
        magic, thash, status, rsv, words, canary = BLOB_HEADER.unpack(out_data[:BLOB_HEADER.size])
        if status != 0:
            result = (None, f"runner status={status}")
            _o0_cache[cache_key] = result
            return result

        body = out_data[BLOB_HEADER.size:-8]
        ohash = hashlib.sha256(body).hexdigest()
        result = (ohash, None)
        _o0_cache[cache_key] = result
        return result


def parse_ptx_inline_instructions(ptx_text: str) -> tuple[list[str], list[str], list[str]]:
    """
    Parse the full PTX, extracting:
    - preamble: everything before the inline asm block
    - inline_body: instructions inside the inline asm block (body only, no setup/teardown)
    - postamble: everything after the inline asm block
    """
    lines = ptx_text.splitlines()
    preamble = []
    inline_decls = []       # .reg declarations inside asm
    inline_setup = []       # initial movs that set up registers from inputs
    inline_body = []        # the computation instructions
    inline_teardown = []    # final movs that write outputs
    postamble = []

    state = "pre"
    in_asm = False
    preamble_includes_asm_start = False

    for line in lines:
        if 'begin inline asm' in line:
            state = "asm_decls"
            preamble.append(line)
            continue
        if 'end inline asm' in line:
            state = "post"
            postamble.append(line)
            continue

        if state == "pre":
            preamble.append(line)
        elif state == "post":
            postamble.append(line)
        elif state in ("asm_decls", "asm_setup", "asm_body", "asm_teardown"):
            stripped = line.strip()
            if not stripped:
                continue

            if state == "asm_decls":
                if stripped.startswith('.reg '):
                    inline_decls.append(stripped)
                elif (stripped.startswith('mov.b32 %xr') or stripped.startswith('mov.u64 %xd') or
                      stripped.startswith('mov.u64 ') and ('0x' in stripped)):
                    inline_setup.append(stripped)
                    state = "asm_setup"
                elif (stripped.startswith('setp.') or stripped.startswith('lop3.') or
                      stripped.startswith('rem.') or stripped.startswith('xor.') or
                      stripped.startswith('add.') or stripped.startswith('sub.') or
                      stripped.startswith('mul.') or stripped.startswith('div.') or
                      stripped.startswith('min.') or stripped.startswith('max.') or
                      stripped.startswith('abs.') or stripped.startswith('neg.') or
                      stripped.startswith('not.') or stripped.startswith('and.') or
                      stripped.startswith('or.') or stripped.startswith('shl.') or
                      stripped.startswith('shr.') or stripped.startswith('shf.') or
                      stripped.startswith('bfe.') or stripped.startswith('bfi.') or
                      stripped.startswith('bfind.') or stripped.startswith('brev.') or
                      stripped.startswith('clz.') or stripped.startswith('cnot.') or
                      stripped.startswith('cvt.') or stripped.startswith('popc.') or
                      stripped.startswith('sad.') or stripped.startswith('mad.') or
                      stripped.startswith('mad24.') or stripped.startswith('selp.') or
                      stripped.startswith('slct.') or stripped.startswith('set.')):
                    inline_body.append(stripped)
                    state = "asm_body"
                else:
                    inline_setup.append(stripped)
                    state = "asm_setup"
            elif state == "asm_setup":
                if (stripped.startswith('setp.') or stripped.startswith('lop3.') or
                    stripped.startswith('rem.') or stripped.startswith('xor.') or
                    stripped.startswith('add.') or stripped.startswith('sub.') or
                    stripped.startswith('mul.') or stripped.startswith('div.') or
                    stripped.startswith('min.') or stripped.startswith('max.') or
                    stripped.startswith('abs.') or stripped.startswith('neg.') or
                    stripped.startswith('not.') or stripped.startswith('and.') or
                    stripped.startswith('or.') or stripped.startswith('shl.') or
                    stripped.startswith('shr.') or stripped.startswith('shf.') or
                    stripped.startswith('bfe.') or stripped.startswith('bfi.') or
                    stripped.startswith('bfind.') or stripped.startswith('brev.') or
                    stripped.startswith('clz.') or stripped.startswith('cnot.') or
                    stripped.startswith('cvt.') or stripped.startswith('popc.') or
                    stripped.startswith('sad.') or stripped.startswith('mad.') or
                    stripped.startswith('mad24.') or stripped.startswith('selp.') or
                    stripped.startswith('slct.') or stripped.startswith('set.')):
                    inline_body.append(stripped)
                    state = "asm_body"
                else:
                    inline_setup.append(stripped)
            elif state == "asm_body":
                if stripped.startswith('mov.u64 %rd1') or stripped.startswith('mov.u64 %rd2'):
                    inline_teardown.append(stripped)
                    state = "asm_teardown"
                else:
                    inline_body.append(stripped)
            elif state == "asm_teardown":
                if stripped.startswith('mov.u64 %rd1') or stripped.startswith('mov.u64 %rd2'):
                    inline_teardown.append(stripped)
                else:
                    inline_body.append(stripped)
                    state = "asm_body"

    return preamble, inline_decls, inline_setup, inline_body, inline_teardown, postamble


def rebuild_ptx(preamble, decls, setup, body, teardown, postamble) -> str:
    """Rebuild the full PTX from components."""
    asm_lines = []
    asm_lines.extend(decls)
    asm_lines.extend(setup)
    asm_lines.extend(body)
    asm_lines.extend(teardown)

    result = []
    for line in preamble:
        result.append(line)
        if 'begin inline asm' in line:
            result.extend(asm_lines)
    # Skip the original asm body lines (they're between begin and end markers)
    result.extend(postamble)
    return '\n'.join(result) + '\n'


def check_divergence(body_instructions: list[str]) -> tuple[bool, str, str]:
    """Test if body_instructions trigger the O0 vs O1 divergence on CUDA 11.3."""
    global preamble, decls, setup, teardown, postamble
    ptx = rebuild_ptx(preamble, decls, setup, body_instructions, teardown, postamble)

    # Check both O0 and O1
    o0_hash, o0_err = compile_and_run(ptx, 0, PTXAS_113)
    if o0_hash is None:
        return False, f"O0:{o0_err}", ""

    o1_hash, o1_err = compile_and_run(ptx, 1, PTXAS_113)
    if o1_hash is None:
        return False, f"O1:{o1_err}", ""

    diverges = (o0_hash != o1_hash)
    return diverges, o0_hash[:16], o1_hash[:16]


def main():
    global preamble, decls, setup, teardown, postamble

    # Load original PTX
    orig_ptx_path = CASE_DIR / "builds" / "cuda-11.3__O0" / "kernel.ptx"
    orig_ptx = orig_ptx_path.read_text()

    # Parse
    preamble, decls, setup, body, teardown, postamble = parse_ptx_inline_instructions(orig_ptx)
    print(f"Parsed PTX:")
    print(f"  Preamble: {len(preamble)} lines")
    print(f"  Decls: {len(decls)} lines")
    print(f"  Setup: {len(setup)} lines")
    print(f"  Body: {len(body)} instructions")
    print(f"  Teardown: {len(teardown)} lines")
    print(f"  Postamble: {len(postamble)} lines")

    # Show body instructions
    print(f"\nBody instructions ({len(body)}):")
    for i, instr in enumerate(body):
        print(f"  [{i:3d}] {instr}")

    # Verify full set reproduces bug
    print(f"\n--- Verifying full set ({len(body)} instructions) ---")
    diverges, h0, h1 = check_divergence(body)
    print(f"Diverges: {diverges} (O0={h0}, O1={h1})")
    if not diverges:
        print("ERROR: Full set does not reproduce bug!")
        sys.exit(1)

    # Verify O1 with 11.8 does NOT diverge (it should match O0)
    ptx = rebuild_ptx(preamble, decls, setup, body, teardown, postamble)
    o0_118_hash, _ = compile_and_run(ptx, 0, PTXAS_118)
    o1_118_hash, _ = compile_and_run(ptx, 1, PTXAS_118)
    print(f"11.8 O0==O1: {o0_118_hash == o1_118_hash} (expected True)")

    # Now bisect
    print(f"\n=== Starting bisection ({len(body)} instructions) ===")
    current = list(body)
    min_size = 2
    step = 0

    while len(current) > min_size:
        step += 1
        half = len(current) // 2

        # Strategy: try second half first (more likely to contain interesting instructions)
        candidate = current[half:]
        diverges, h0, h1 = check_divergence(candidate)
        pct = len(candidate) / len(body) * 100
        print(f"  Step {step}: [{len(candidate)}/{len(body)} ({pct:.0f}%)] "
              f"(second half) diverges={diverges} O0={h0} O1={h1}")

        if diverges:
            current = candidate
            print(f"    -> KEEP second half")
            continue

        # Try first half
        candidate = current[:half]
        diverges, h0, h1 = check_divergence(candidate)
        pct = len(candidate) / len(body) * 100
        print(f"  Step {step}b: [{len(candidate)}/{len(body)} ({pct:.0f}%)] "
              f"(first half) diverges={diverges} O0={h0} O1={h1}")

        if diverges:
            current = candidate
            print(f"    -> KEEP first half")
            continue

        # Neither half alone triggers the bug
        # Try removing individual instructions from the back
        print(f"    -> Neither half triggers alone, attempting single-instruction removal")
        found = False
        for i in range(len(current) - 1, -1, -1):
            test = current[:i] + current[i+1:]
            diverges, h0, h1 = check_divergence(test)
            if diverges:
                current = test
                print(f"    -> Removed instruction [{i}], now {len(current)} instrs")
                found = True
                break
        if not found:
            # Try from the front
            for i in range(len(current)):
                test = current[:i] + current[i+1:]
                diverges, h0, h1 = check_divergence(test)
                if diverges:
                    current = test
                    print(f"    -> Removed instruction [{i}], now {len(current)} instrs")
                    found = True
                    break
        if not found:
            print(f"    -> No single instruction can be removed, at minimum")
            break

    print(f"\n=== Minimization complete ===")
    print(f"Original: {len(body)} instructions")
    print(f"Minimized: {len(current)} instructions")
    print(f"Reduction: {(1 - len(current)/len(body))*100:.0f}%")

    # Save minimized PTX
    min_ptx = rebuild_ptx(preamble, decls, setup, current, teardown, postamble)
    out_path = CASE_DIR / "minimized_repro.ptx"
    out_path.write_text(min_ptx)
    print(f"\nMinimized PTX saved to: {out_path}")

    # Print the minimized instructions
    print(f"\nMinimized body instructions:")
    for i, instr in enumerate(current):
        print(f"  [{i}] {instr}")

    # Save report
    ops = set()
    for instr in current:
        m = re.match(r'^(\w+)\.', instr)
        if m:
            ops.add(m.group(1))
    report = {
        "original_count": len(body),
        "minimized_count": len(current),
        "instructions": current,
        "opcodes": sorted(ops),
    }
    report_path = CASE_DIR / "minimization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {report_path}")


# Make these module-level so check_divergence can access them
preamble = []
decls = []
setup = []
teardown = []
postamble = []

if __name__ == "__main__":
    main()
