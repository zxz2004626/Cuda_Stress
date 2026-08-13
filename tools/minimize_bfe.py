#!/usr/bin/env python3
"""
Focused minimizer for inline_ptx__4e889aef7c32d967.

The bug is CUDA 11.3 ptxas -O1 specific. Key suspicious instructions:
  bfe.s32 with pos+len > 32 (over-MSB, well-defined per PTX spec but historically buggy)

This script:
1. Extracts the 3 bfe instructions from the PTX
2. Tests each in isolation with ptxas 11.3 O0 vs O1
3. Produces a minimal standalone .ptx reproducer
"""

import subprocess, sys, os, json, tempfile, hashlib, struct
from pathlib import Path

PTXAS_113 = "/usr/local/cuda-11.3/bin/ptxas"
CUDA_RUNNER = "/home/linux/cuda_stress/native/cuda_runner"

# Minimal PTX that tests bfe over-MSB behavior
# We feed known input values and check if O0 and O1 agree
MINIMAL_PTX = """\
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
    .reg .b32 %r<20>;
    .reg .b64 %rd<20>;

    ld.param.u64 %rd5, [test_kernel_param_0];
    ld.param.u64 %rd6, [test_kernel_param_1];

    // Test vector: %r1 = 0x80000000 (bit 31 set), %r2 = 0xFFFFFFFF (all bits set)
    mov.u32 %r1, 0x80000000;
    mov.u32 %r2, 0xffffffff;

    // ---- bfe over-MSB cases ----

    // Case 1: pos=31, len=1 (boundary extraction of sign bit)
    // Expected: extract bit 31 of %r2 (=1), sign-extend to 32 bits → 0xffffffff
    bfe.s32 %r3, %r2, 31, 1;

    // Case 2: pos=31, len=32 (over-MSB: pos+len=63 > 32)
    // Expected: extract bits [31,62] of %r1. Only bit 31 exists (bit[31]=1).
    // Bits 32-62 are beyond register → 0. Extracted field = {31'b0, bit[31]} = 1.
    // Sign-extended from bit 31 of extracted field (which is 0) → 1.
    bfe.s32 %r4, %r1, 31, 32;

    // Case 3: pos=32, len=1 (pos beyond register width)
    // Expected: extract bit 32 of %r2. Beyond register → 0. Result = 0.
    bfe.s32 %r5, %r2, 32, 1;

    // Case 4: pos=32, len=1 on %r1 which has bit 31=1
    // Same as case 3 but different source → should also be 0
    bfe.s32 %r6, %r1, 32, 1;

    // ---- Mix bfe results into outputs ----
    cvt.u64.u32 %rd1, %r3;
    cvt.u64.u32 %rd2, %r4;
    xor.b64 %rd1, %rd1, %rd2;
    cvt.u64.u32 %rd2, %r5;
    xor.b64 %rd1, %rd1, %rd2;
    cvt.u64.u32 %rd2, %r6;
    xor.b64 %rd1, %rd1, %rd2;

    // Output: %rd1 should be deterministic (xor of 4 expected values)
    mov.u64 %rd2, %rd5;
    st.global.u64 [%rd6], %rd1;
    st.global.u64 [%rd6+8], %rd2;
    ret;
}
"""

# More complex test with bfe embedded in a computation chain
BODY_PTX = """\
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
    .reg .pred %p<8>;

    ld.param.u64 %rd5, [test_kernel_param_0];
    ld.param.u64 %rd6, [test_kernel_param_1];
    ld.param.u32 %r5, [test_kernel_param_2];
    ld.param.u64 %rd7, [test_kernel_param_3];

    mov.u64 %rd3, %rd5;
    mov.u64 %rd4, %rd6;
    mov.u32 %r1, 0x80000000;
    mov.u32 %r2, 0xffffffff;
    mov.u32 %r3, 0x12345678;
    mov.u32 %r4, 0x87654321;
    mov.u64 %xd0, %rd3;
    mov.u64 %xd1, %rd4;
    mov.u64 %xd2, 0x8d605920aa50f237;
    mov.u64 %xd3, 0xe8c94397436b9a06;
    mov.u64 %xd4, 0x65c05d0731bcaa11;
    mov.u64 %xd5, 0xad81107e5ebc92cf;
    setp.eq.u32 %xp0, %r4, %r4;
    setp.ne.u32 %xp1, %r1, %r2;
    setp.lt.u32 %xp2, %r1, %r2;
    setp.eq.u32 %xp3, %r1, %r4;

    {body}

    mov.u64 %rd1, %xd0;
    mov.u64 %rd2, %xd1;
    st.global.u64 [%rd6], %rd1;
    st.global.u64 [%rd6+8], %rd2;
    ret;
}
"""


def compile_and_run(ptx_text: str, opt_level: int) -> bytes | None:
    """Compile PTX with ptxas 11.3 at given opt level and run on GPU. Returns output bytes or None."""
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        ptx_path = tdir / "kernel.ptx"
        cubin_path = tdir / "kernel.cubin"
        ptx_path.write_text(ptx_text)

        # Compile
        r = subprocess.run([PTXAS_113, "-arch=sm_86", f"-O{opt_level}",
                            "-o", str(cubin_path), str(ptx_path)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ptxas -O{opt_level} FAILED: {r.stderr[:200]}")
            return None

        # Run on GPU
        # Prepare input: 256 u64 words (2048 bytes), first word = 0xdeadbeef...
        input_data = struct.pack('<Q', 0xDEADBEEFCAFEBABE) * 256
        input_path = tdir / "input.bin"
        input_path.write_bytes(input_data)

        r = subprocess.run([CUDA_RUNNER, str(cubin_path), str(input_path),
                            "256", "64", "1", "1", "0", "8", "2"],
                          capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"  runner FAILED: {r.stderr[:200]}")
            return None

        # Parse runner output to get result bytes
        # The runner writes a blob; we need to extract the output words
        output_path = tdir / "output.bin"
        # Runner output format: we need to look at what the runner actually does
        # Actually, let me check the runner's expected interface

        return None  # FIXME


def main():
    # For now, just compile and check SASS differences
    print("=== Testing bfe over-MSB with ptxas 11.3 ===")

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        ptx_path = tdir / "kernel.ptx"
        ptx_path.write_text(MINIMAL_PTX)

        for opt in [0, 1]:
            cubin_path = tdir / f"kernel_o{opt}.cubin"
            r = subprocess.run([PTXAS_113, "-arch=sm_86", f"-O{opt}",
                                "-o", str(cubin_path), str(ptx_path)],
                              capture_output=True, text=True)
            print(f"  ptxas -O{opt}: {'OK' if r.returncode == 0 else 'FAIL'} "
                  f"({len(cubin_path.read_bytes()) if cubin_path.exists() else 0} bytes)")
            if r.returncode != 0:
                print(f"    stderr: {r.stderr[:300]}")

    # Now test with CUDA 11.8 ptxas for comparison
    print("\n=== Testing same PTX with ptxas 11.8 (reference) ===")
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        ptx_path = tdir / "kernel.ptx"
        ptx_path.write_text(MINIMAL_PTX)

        for opt in [0, 1]:
            cubin_path = tdir / f"kernel_o{opt}.cubin"
            r = subprocess.run(["/usr/local/cuda-11.8/bin/ptxas", "-arch=sm_86",
                                f"-O{opt}", "-o", str(cubin_path), str(ptx_path)],
                              capture_output=True, text=True)
            print(f"  ptxas -O{opt}: {'OK' if r.returncode == 0 else 'FAIL'} "
                  f"({len(cubin_path.read_bytes()) if cubin_path.exists() else 0} bytes)")

    # Also test the original bug's PTX directly through ptxas 11.3 O0 vs O1
    # to see if we can reproduce at the PTX level
    print("\n=== Reproducing original bug at PTX level ===")
    orig_ptx = Path("/home/linux/cuda_stress/runs/run-20260812-115625-1149/differentials/inline_ptx__4e889aef7c32d967/builds/cuda-11.3__O0/kernel.ptx").read_text()

    # Compile original PTX with 11.3 O0 and O1, run on GPU, compare outputs
    for opt in [0, 1]:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            ptx_path = tdir / "kernel.ptx"
            cubin_path = tdir / "kernel.cubin"
            ptx_path.write_text(orig_ptx)
            r = subprocess.run([PTXAS_113, "-arch=sm_86", f"-O{opt}",
                                "-o", str(cubin_path), str(ptx_path)],
                              capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  Original PTX ptxas -O{opt}: FAILED {r.stderr[:200]}")
            else:
                # Disassemble
                r2 = subprocess.run(["/usr/local/cuda-11.3/bin/nvdisasm", "-ndf", "-g", str(cubin_path)],
                                   capture_output=True, text=True)
                lines = r2.stdout.count('\n')
                print(f"  Original PTX ptxas -O{opt}: {cubin_path.stat().st_size} bytes CUBIN, ~{lines} SASS lines")


if __name__ == "__main__":
    main()
