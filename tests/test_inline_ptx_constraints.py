"""PTX ISA 9.3 constraint and coverage checks for the inline PTX generator."""

from pathlib import Path

from generator.strategies.inline_ptx import (
    M32,
    M64,
    OPS,
    _bfe,
    _bfi_bits,
    _bmsk,
    _gen_program,
    _prmt,
    _shl,
    _shr,
)
from tools.audit_ptx_coverage import coverage_report


def _compute_steps(program):
    return [step for step in program.steps if step["op"] != "out"]


def test_plain_shift_counts_clamp_instead_of_wrap():
    # PTX ISA 9.3 sections 9.7.8.8/9: counts > N clamp to N.
    assert _shl(1, 32, 32) == 0
    assert _shl(1, 33, 32) == 0
    assert _shr(M32, 32, 32) == 0
    assert _shr(M32, 32, 32, signed=True) == M32
    assert _shl(1, 64, 64) == 0
    assert _shr(M64, 65, 64) == 0


def test_bfe_bfi_cover_zero_and_cross_msb_fields():
    assert _bfe(M32, 0, 0, 32, False) == 0
    assert _bfe(0x80000000, 31, 8, 32, False) == 1
    assert _bfe(0x80000000, 31, 8, 32, True) == M32
    assert _bfe(M32, 32, 8, 32, False) == 0
    assert _bfi_bits(0, 0x12345678, 0, 0, 32) == 0x12345678
    assert _bfi_bits(M32, 0, 7, 39, 32) == 0xFFFFFF80
    assert _bfi_bits(M64, 0, 13, 71, 64) == (M64 << 13) & M64


def test_prmt_generic_selector_uses_full_nibble():
    # Selector 8 chooses byte 0 and replicates its sign bit; the old generator
    # incorrectly restricted every selector to 0..7.
    assert _prmt(0x00000080, 0, 0x0008) & 0xFF == 0xFF
    assert _prmt(0x0000007F, 0, 0x0008) & 0xFF == 0


def test_bmsk_modes_follow_clamp_and_wrap_rules():
    assert _bmsk(7, 39, "clamp") == 0xFFFFFF80
    assert _bmsk(32, 1, "clamp") == 0
    assert _bmsk(7, 39, "wrap") == 0x00003F80
    assert _bmsk(0, 32, "wrap") == 0


def test_64_bit_shift_count_operand_is_r32():
    shifts = [op for op in OPS if op["name"] in ("shl.b64", "shr.u64", "shr.s64")]
    assert shifts
    assert all(op["srcs"] == ["r64", "r32"] for op in shifts)


def test_isa_and_sm_requirements_gate_newer_variants():
    common = _gen_program(7, {
        "n_instr": 240, "n_regs32": 12, "n_regs64": 8, "n_preds": 6,
        "use_rare_ops": True, "use_immediates": True,
        "coverage_mode": "balanced", "max_ptx_isa": "7.3", "target_sm": "sm_86",
    })
    newer = _gen_program(7, {
        "n_instr": 240, "n_regs32": 12, "n_regs64": 8, "n_preds": 6,
        "use_rare_ops": True, "use_immediates": True,
        "coverage_mode": "balanced", "max_ptx_isa": "7.6", "target_sm": "sm_86",
    })
    assert not any(step["op"].startswith("bmsk.") for step in common.steps)
    assert any(step["op"].startswith("bmsk.") for step in newer.steps)


def test_balanced_mode_covers_every_eligible_variant_and_boundary_class():
    params = {
        "n_instr": 240, "n_regs32": 12, "n_regs64": 8, "n_preds": 6,
        "use_rare_ops": True, "use_immediates": True,
        "coverage_mode": "balanced", "max_ptx_isa": "7.3", "target_sm": "sm_86",
    }
    program = _gen_program(12345, params)
    names = {step["op"] for step in _compute_steps(program)}
    eligible = {
        op["name"] for op in OPS
        if op["min_ptx"] <= (7, 3) and op["min_sm"] <= 86
    }
    assert eligible <= names

    bfi32 = [step["imm"] for step in program.steps if step["op"] == "bfi.b32"]
    prmt = [step["imm"] for step in program.steps if step["op"] == "prmt.b32"]
    assert (7, 39) in bfi32
    assert any(any((selector >> (4 * i)) & 8 for i in range(4)) for selector in prmt)

    # Every generated bfe/bfi immediate remains inside the documented u8 range.
    for step in program.steps:
        if step["op"].startswith(("bfe.", "bfi.")):
            assert all(0 <= value <= 255 for value in step["imm"])


def test_document_coverage_audit_recognizes_all_generator_mnemonics():
    report = coverage_report(
        Path(__file__).parents[1] / "ptx_isa_reference" / "headings.tsv"
    )
    assert report["documented_instruction_families"] >= 120
    assert report["inline_ptx_variants"] == len(OPS)
    assert report["generator_only_names"] == []
    assert report["covered_instruction_families"] >= 45
    assert report["covered_section_families"] >= 45
