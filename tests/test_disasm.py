"""SASS / PTX parsing and normalized hashes."""

import hashlib

from pipeline import disasm

SASS_SAMPLE = """\
\tcode for sm_86
\t\tFunction : test_kernel
.headerflags\t@"EF_CUDA_TEXMODE_UNIFIED"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   ISETP.LT.U32.AND P0, PT, R5, RZ, PT ;
        /*0020*/        @!P0       BRA `(0x40) ;
        /*0030*/                   LDG.E R8, [R2.64] ;
        /*0040*/                   STG.E [R4.64], R8 ;
        /*0050*/                   BSSY B1, `(0x70) ;
"""

PTX_SAMPLE = """\
.version 8.5
.target sm_86
.address_size 64
.visible .entry test_kernel( .param .u64 input, .param .u64 output, .param .u32 n, .param .u64 seed ) {
  .reg .b32 %r<10>;
  ld.param.u64 %rd1, [input];
  ld.param.u64 %rd2, [output];
  mov.u32 %r1, 0;
  isetp.lt.u32.and p0, %r1, %r2, p0;
  @p0 bra L0;
  mul.wide.u32 %rd3, %r1, %r2;
  add.s64 %rd4, %rd3, %rd1;
  st.global.u64 [%rd2], %rd4;
L0:
  ret;
}
"""


def test_parse_sass_counts():
    m = disasm.parse_sass(SASS_SAMPLE)
    assert m["instruction_count"] == 6
    assert m["branch_count"] == 2          # BRA + BSSY
    assert m["load_count"] == 1
    assert m["store_count"] == 1
    assert m["predicated_count"] == 1      # @!P0 BRA
    assert "IMAD" in m["opcode_histogram"]
    assert "LDG" in m["opcode_histogram"]


def test_sass_normalized_hash_canonicalizes_registers():
    s1 = disasm.parse_sass(SASS_SAMPLE)["normalized_hash"]
    # Same instructions but different register numbers -> same normalized hash.
    renamed = SASS_SAMPLE.replace("R1", "R9").replace("R2", "R7").replace("R4", "R3").replace("R5", "R6").replace("R8", "R2")
    s2 = disasm.parse_sass(renamed)["normalized_hash"]
    assert s1 == s2


def test_sass_normalized_hash_sensitive_to_schedule():
    # Removing an instruction must change the hash.
    trimmed = SASS_SAMPLE.replace("LDG.E R8, [R2.64] ;", "")
    assert (disasm.parse_sass(SASS_SAMPLE)["normalized_hash"]
            != disasm.parse_sass(trimmed)["normalized_hash"])


def test_parse_ptx():
    m = disasm.parse_ptx(PTX_SAMPLE)
    assert m["instruction_count"] == 8
    assert m["predicated_count"] == 1
    assert "isetp" in m["opcode_histogram"] or "isetp" in m["opcode_histogram"]
    assert "mul" in m["opcode_histogram"]


def test_parse_ptxas_verbose():
    v = "ptxas info    : Used 40 registers, used 0 barriers, 384 bytes cmem[0]\n" \
        "ptxas info    : 0 bytes stack frame, 16 bytes spill stores, 8 bytes spill loads"
    m = disasm.parse_ptxas_verbose(v)
    assert m["registers"] == 40
    assert m["stack_frame_bytes"] == 0
    assert m["spill_stores_bytes"] == 16
    assert m["spill_loads_bytes"] == 8
