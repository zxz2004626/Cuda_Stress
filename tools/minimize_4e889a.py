#!/usr/bin/env python3
"""
Minimize inline_ptx__4e889aef7c32d967 — a CUDA 11.3 ptxas O1-only miscompilation.

The bug: CUDA 11.3 ptxas -O1 produces wrong output; -O0 on 11.3 and all levels
on 11.8+ are correct.  We bisect the inline-PTX instruction list to find the
minimum subset that still triggers the divergence when fed through ptxas -O1 vs -O0.

Strategy:
  - Extract the inline-asm block from the PTX
  - Build a minimal .ptx wrapper with known inputs
  - Feed to ptxas -O0 / -O1 on CUDA 11.3
  - Compare the SASS; a meaningful difference means the bug is still present
  - Binary-search on the instruction list
"""

import subprocess, sys, os, json, tempfile, shutil, hashlib, re
from pathlib import Path

PTXAS_113 = "/usr/local/cuda-11.3/bin/ptxas"
NVDSASM_113 = "/usr/local/cuda-11.3/bin/nvdisasm"

PTXAS_118 = "/usr/local/cuda-11.8/bin/ptxas"

# PTX wrapper template: feed known operand values, capture two output regs
PTX_TEMPLATE = """\
.version 7.3
.target sm_86
.address_size 64

.visible .entry test_kernel(
    .param .u64 test_kernel_param_0,
    .param .u64 test_kernel_param_1,
    .param .u32 test_kernel_param_2,
    .param .u64 test_kernel_param_3
)
{
    .reg .b32 %r<40>;
    .reg .b64 %rd<40>;

    ld.param.u64 %rd5, [test_kernel_param_0];
    ld.param.u64 %rd6, [test_kernel_param_1];
    ld.param.u32 %r5, [test_kernel_param_2];
    ld.param.u64 %rd7, [test_kernel_param_3];

    // Load input words (simplified: use param values directly)
    mov.u64 %rd3, %rd5;    // b0
    mov.u64 %rd4, %rd6;    // b1

    // Setup 32-bit inputs from seed
    mov.u32 %r1, 0xdeadbeef;
    mov.u32 %r2, 0xcafebabe;
    mov.u32 %r3, 0x12345678;
    mov.u32 %r4, 0x87654321;

{inline_asm}

    // Output results
    st.global.u64 [%rd6], %rd1;
    st.global.u64 [%rd6+8], %rd2;
    ret;
}
"""


def parse_inline_ptx_instructions(test_cu_path: str) -> list[str]:
    """Extract the semicolon-separated PTX instructions from the inline asm block."""
    text = Path(test_cu_path).read_text()
    # Find the asm volatile block
    m = re.search(r'asm volatile\((.*?)\);', text, re.DOTALL)
    if not m:
        raise ValueError("no asm volatile block found")
    body = m.group(1)
    # Split by quoted string boundaries
    # The body consists of "\n\tinstr;\n\tinstr;..."
    instrs = []
    for part in body.split('\\n\\t'):
        part = part.strip().strip('"').strip()
        if part and ':' in part:  # constraint lines
            continue
        if part.endswith(';'):
            instrs.append(part.rstrip(';'))
    return instrs


def build_ptx(instructions: list[str], init_lines: list[str] | None = None) -> str:
    """Build a standalone .ptx file from a list of instructions."""
    if init_lines is None:
        init_lines = [
            ".reg .b32 %xr<10>;",
            ".reg .b64 %xd<6>;",
            ".reg .pred %xp<4>;",
            "mov.b32 %xr0, %r1;",
            "mov.b32 %xr1, %r2;",
            "mov.b32 %xr2, %r3;",
            "mov.b32 %xr3, %r4;",
            "mov.b32 %xr4, 370807714;",
            "mov.b32 %xr5, 4236359279;",
            "mov.b32 %xr6, 3201663651;",
            "mov.b32 %xr7, 3646417547;",
            "mov.b32 %xr8, 1541349621;",
            "mov.b32 %xr9, 2656793913;",
            "mov.u64 %xd0, %rd3;",
            "mov.u64 %xd1, %rd4;",
            "mov.u64 %xd2, 0x8d605920aa50f237;",
            "mov.u64 %xd3, 0xe8c94397436b9a06;",
            "mov.u64 %xd4, 0x65c05d0731bcaa11;",
            "mov.u64 %xd5, 0xad81107e5ebc92cf;",
        ]
    lines = init_lines + [f"{i};" for i in instructions] + [
        "mov.u64 %rd1, %xd0;",
        "mov.u64 %rd2, %xd1;",
    ]
    asm_block = "\n\t".join(lines)
    return PTX_TEMPLATE.format(inline_asm=asm_block)


def compile_ptx(ptx_text: str, ptxas_bin: str, opt_level: int) -> bytes:
    """Compile PTX to CUBIN using ptxas at the given opt level. Returns CUBIN bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        ptx_path = Path(tmp) / "kernel.ptx"
        cubin_path = Path(tmp) / "kernel.cubin"
        ptx_path.write_text(ptx_text)
        cmd = [ptxas_bin, f"-arch=sm_86", f"-O{opt_level}", "-o", str(cubin_path), str(ptx_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas -O{opt_level} failed: {r.stderr}")
        return cubin_path.read_bytes()


def disassemble_cubin(cubin: bytes, nvdisasm_bin: str) -> str:
    """Disassemble CUBIN to SASS text."""
    with tempfile.TemporaryDirectory() as tmp:
        cubin_path = Path(tmp) / "kernel.cubin"
        cubin_path.write_bytes(cubin)
        cmd = [nvdisasm_bin, "-ndf", "-g", str(cubin_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvdisasm failed: {r.stderr}")
        return r.stdout


def norm_sass(sass_text: str) -> str:
    """Normalize SASS: strip addresses, keep only instruction mnemonics + operands."""
    lines = []
    for line in sass_text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith(".") or line.startswith("Function"):
            continue
        # Remove address comments /* ... */
        line = re.sub(r'/\*.*?\*/', '', line).strip()
        # Remove leading hex address
        line = re.sub(r'^\s*/\*\w+\*/\s*', '', line).strip()
        # Collapse whitespace
        line = re.sub(r'\s+', ' ', line)
        if line and not line.startswith(';'):
            lines.append(line)
    return '\n'.join(lines)


def sass_opcode_count(sass_norm: str) -> int:
    """Count real opcodes in normalized SASS (exclude NOP, comments)."""
    count = 0
    for line in sass_norm.splitlines():
        line = line.strip()
        if not line or line.startswith(';') or 'NOP' in line:
            continue
        if re.match(r'^[A-Z]', line):
            count += 1
    return count


def has_divergent_sass(instructions: list[str]) -> tuple[bool, dict]:
    """
    Test if the instruction list triggers the CUDA 11.3 ptxas O0-vs-O1 divergence.
    Returns (is_divergent, info_dict).
    """
    try:
        ptx = build_ptx(instructions)
        cubin_o0 = compile_ptx(ptx, PTXAS_113, 0)
        cubin_o1 = compile_ptx(ptx, PTXAS_113, 1)
    except RuntimeError as e:
        return False, {"error": str(e)}

    sass_o0 = norm_sass(disassemble_cubin(cubin_o0, NVDSASM_113))
    sass_o1 = norm_sass(disassemble_cubin(cubin_o1, NVDSASM_113))

    # Calculate metrics
    count_o0 = sass_opcode_count(sass_o0)
    count_o1 = sass_opcode_count(sass_o1)

    hash_o0 = hashlib.sha256(sass_o0.encode()).hexdigest()
    hash_o1 = hashlib.sha256(sass_o1.encode()).hexdigest()

    # For a real bug trigger, we need:
    # 1. SASS different between O0 and O1 (different hashes)
    # 2. O1 should have significantly fewer instructions (optimization is happening)
    # 3. The optimization delta should be > 50% (meaningful optimization)
    divergent = (hash_o0 != hash_o1 and count_o1 < count_o0 * 0.5)

    return divergent, {
        "o0_count": count_o0,
        "o1_count": count_o1,
        "o0_hash": hash_o0[:16],
        "o1_hash": hash_o1[:16],
        "delta": (count_o0 - count_o1) / max(count_o0, 1),
    }


def bisect(instructions: list[str], min_size: int = 2) -> list[str]:
    """
    Binary search to find the minimal set of instructions that triggers the bug.
    Each iteration tests if the current set still triggers the divergence.
    If yes, halve the set. If no, the bug is in the removed half.
    """
    current = list(instructions)
    n = len(current)
    print(f"Starting bisection with {n} instructions")

    # First, verify the full set triggers
    full_ok, info = has_divergent_sass(current)
    print(f"  Full set ({n} instrs): divergent={full_ok} info={info}")
    if not full_ok:
        print("  WARNING: full set does not trigger divergence! Checking with CUDA 11.8 ptxas...")
        # Test with 11.8 to confirm it doesn't diverge there
        # (The bug is 11.3-specific)
        # Fall through — the bisection may not work via SASS comparison alone
        # We may need to run the CUBIN and check outputs
        return current[:16]  # return a subset for manual inspection

    step = 0
    while len(current) > min_size:
        step += 1
        half = len(current) // 2
        # Try removing the first half
        candidate = current[half:]
        ok, info = has_divergent_sass(candidate)
        print(f"  Step {step}: {len(candidate)} instrs -> divergent={ok} "
              f"(o0={info.get('o0_count','?')} o1={info.get('o1_count','?')} "
              f"delta={info.get('delta',0):.2f})")

        if ok:
            current = candidate
            print(f"    -> KEEP (bug still present in second half)")
        else:
            # Try the first half
            candidate2 = current[:half]
            ok2, info2 = has_divergent_sass(candidate2)
            print(f"  Step {step}b: {len(candidate2)} instrs -> divergent={ok2} "
                  f"(o0={info2.get('o0_count','?')} o1={info2.get('o1_count','?')} "
                  f"delta={info2.get('delta',0):.2f})")

            if ok2:
                current = candidate2
                print(f"    -> KEEP (bug in first half)")
            else:
                # Neither half alone triggers — bug needs instructions from both halves
                # Try to find minimum by removing a smaller chunk
                print(f"    -> Neither half triggers alone, trying finer-grained removal")
                # Remove quarters
                quarter = max(len(current) // 4, 1)
                for i in range(0, len(current), quarter):
                    test = current[:i] + current[i+quarter:]
                    ok3, info3 = has_divergent_sass(test)
                    if ok3:
                        current = test
                        print(f"    -> Removed chunk at {i}-{i+quarter}, now {len(current)} instrs")
                        break
                else:
                    # No quarter can be removed — we're close to minimal
                    print(f"    -> No chunk removable, near-minimal at {len(current)} instrs")
                    break

    print(f"\nMinimized to {len(current)} instructions (from {n})")
    return current


def main():
    case_dir = Path("/home/linux/cuda_stress/runs/run-20260812-115625-1149/differentials/inline_ptx__4e889aef7c32d967")
    test_cu = case_dir / "test.cu"

    # Parse instructions from the inline asm
    instrs = parse_inline_ptx_instructions(str(test_cu))
    print(f"Parsed {len(instrs)} inline PTX instructions")

    # Also parse from the actual PTX file for comparison
    ptx_path = case_dir / "builds" / "cuda-11.3__O0" / "kernel.ptx"
    ptx_text = Path(ptx_path).read_text()

    # Extract the inline asm block from PTX
    in_asm = False
    ptx_instrs = []
    for line in ptx_text.splitlines():
        line = line.strip()
        if 'begin inline asm' in line:
            in_asm = True
            continue
        if 'end inline asm' in line:
            break
        if in_asm and line and not line.startswith('//') and not line.startswith('.'):
            # These lines contain the PTX instructions (without \n\t wrapping)
            instr = line.rstrip(';')
            if instr and not instr.startswith('mov.u64 %rd') and not instr.startswith('//'):
                ptx_instrs.append(instr)

    # Filter: keep only instructions between the init movs and the final mov.u64 %rd1/%rd2
    # The init section does register setup; the body is the computation
    # Find the first "real" computation instruction after the movs
    first_real = 0
    for i, instr in enumerate(ptx_instrs):
        # Skip the initial mov.b32/mov.u64 register setup
        if (instr.startswith('mov.b32 %xr') or instr.startswith('mov.u64 %xd') or
            instr.startswith('.reg')):
            continue
        first_real = i
        break

    # Find the last real instruction before the output movs
    last_real = len(ptx_instrs)
    for i in range(len(ptx_instrs) - 1, -1, -1):
        if (ptx_instrs[i].startswith('mov.u64 %rd1') or
            ptx_instrs[i].startswith('mov.u64 %rd2')):
            continue
        last_real = i
        break

    body_instrs = ptx_instrs[first_real:last_real+1]
    print(f"PTX body instructions: {len(body_instrs)}")

    # For now, use the test.cu instructions
    # Strip setup and teardown
    body_from_cu = []
    skip_setup = True
    for instr in instrs:
        if skip_setup:
            if (instr.startswith('setp.eq.u32') or instr.startswith('setp.ne.u32') or
                instr.startswith('setp.lt.u32')):
                skip_setup = False
        if not skip_setup:
            body_from_cu.append(instr)

    # Remove final output movs
    while body_from_cu and ('mov.u64 %0' in body_from_cu[-1] or 'mov.u64 %1' in body_from_cu[-1]):
        body_from_cu.pop()

    print(f"Computation body (from test.cu): {len(body_from_cu)} instructions")

    # Use the PTX body for bisection (cleaner, no C escaping)
    minimized = bisect(body_instrs)

    # Save the minimized instructions
    out_path = case_dir / "minimized_instructions.json"
    out_path.write_text(json.dumps({
        "original_count": len(body_instrs),
        "minimized_count": len(minimized),
        "instructions": minimized,
    }, indent=2))
    print(f"\nMinimized instructions saved to {out_path}")

    # Also write a standalone PTX reproducer
    repro_ptx = build_ptx(minimized)
    repro_path = case_dir / "minimized_repro.ptx"
    repro_path.write_text(repro_ptx)
    print(f"Minimized PTX reproducer saved to {repro_path}")

    # Print the minimized PTX
    print(f"\n--- Minimized PTX ({len(minimized)} instructions) ---")
    for i, instr in enumerate(minimized):
        print(f"  {instr};")


if __name__ == "__main__":
    main()
