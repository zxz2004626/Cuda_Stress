#!/usr/bin/env python3
"""Overnight differential analyzer.

Polls a run's results.jsonl for `potential_bug` cases that have not yet been analyzed,
then for each one:
  1. loads the case artifacts,
  2. runs a PTX-ISA UB audit on every inline-asm instruction (against the stored PTX ISA
     9.3 reference semantics: shift counts clamp, bfe/bfi pos/len 0..255, div/rem nonzero
     divisors, prmt selectors 0..15, etc.),
  3. classifies the root cause by op-family signature and compares it with the known
     trigger classes (prmt a==a / over-MSB bfe·bfi / set·bfind / shf),
  4. appends a report entry to runs/<id>/overnight_report.md.

It is designed to run alongside `pipeline.py loop --duration 8h`: it never touches GPU
work beyond re-reading the already-recorded build outputs, so it does not slow the loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from generator.base import default_registry
from generator.strategies.inline_ptx import _gen_program, OPS

REF = pathlib.Path(__file__).resolve().parent.parent / "ptx_isa_reference" / "ptx_isa_9.3.txt"


# ---------------------------------------------------------------------------
# PTX-ISA UB audit
# ---------------------------------------------------------------------------
def _ref_section(opcode: str) -> str:
    """Find the PTX ISA reference text for an opcode (best effort)."""
    if not REF.exists():
        return ""
    key = opcode.split(".")[0]
    m = re.search(rf"\n\s*{re.escape(key)}(?:\s|\n|{{).*?(?=\n\s*[A-Za-z][a-zA-Z0-9_]*\s+(?:\.|{{)|\Z)", REF.read_text(), re.S)
    return m.group(0)[:1200] if m else ""


def audit_step_against_isa(opname: str, imm) -> list[str]:
    """Check a single generated instruction's operands for PTX-ISA violations.

    Returns a list of problems (empty == no UB found).
    """
    problems: list[str] = []
    if opname in ("div.u32_i", "rem.u32_i", "div.u64_i", "rem.u64_i"):
        # divisor is an immediate, always nonzero in the generator; PTX div/rem by zero
        # is UB (or traps) -- assert the immediate is not zero.
        if imm == 0:
            problems.append(f"{opname}: divisor 0 is not permitted")
    elif opname in ("bfe.u32", "bfe.s32", "bfe.u64", "bfe.s64",
                    "bfi.b32", "bfi.b64"):
        if imm is None:
            return problems
        pos, length = imm
        if not (0 <= pos <= 255 and 0 <= length <= 255):
            problems.append(f"{opname}: pos/length outside [0,255]: {imm}")
        # cross-MSB fields are well-defined (zero-extend / drop) -> no extra check needed.
    elif opname == "prmt.b32":
        if imm is not None:
            for k in range(4):
                nib = (imm >> (4 * k)) & 0xF
                # selectors 0..15 are all legal; 0xE/0xF give 0xFF/0x00 per PTX.
                if nib > 15:
                    problems.append(f"prmt: selector nibble {k} = {nib} > 15")
    elif opname in ("shl.b32", "shl.b64", "shr.u32", "shr.u64",
                    "shr.s32", "shr.s64", "shl.b32_i", "shl.b64_i",
                    "shr.u32_i", "shr.u64_i", "shr.s32_i", "shr.s64_i"):
        # PTX clamps counts > width to the width -- any u32 count is legal, no UB.
        pass
    elif opname in ("shf.r.wrap.b32", "shf.r.clamp.b32", "shf.l.wrap.b32", "shf.l.clamp.b32"):
        if imm is not None and not (0 <= imm <= 255):
            problems.append(f"{opname}: shift count {imm} outside [0,255]")
    elif opname == "fns.b32":
        # PTX ISA: "Operand base must be <= 31, otherwise behavior is undefined."
        if imm is not None and not (0 <= imm[0] <= 31):
            problems.append(f"fns: base {imm[0]} outside [0,31]")
    return problems


def ub_audit_program(prog) -> dict:
    """Audit every step of a generated inline-ptx program against the ISA."""
    problems = []
    seen = set()
    for si, st in enumerate(prog.steps):
        if st["op"] == "out":
            continue
        p = audit_step_against_isa(st["op"], st["imm"])
        for x in p:
            problems.append((si, st["op"], st["imm"], x))
        seen.add(st["op"])
    return {"problems": problems, "opcode_count": len(seen), "steps": len(prog.steps)}


# ---------------------------------------------------------------------------
# Root-cause classification
# ---------------------------------------------------------------------------
def op_signature(prog) -> dict:
    from collections import Counter
    c = Counter(st["op"] for st in prog.steps if st["op"] != "out")
    return dict(c)


def classify_root_cause(prog) -> dict:
    sig = op_signature(prog)
    present = set(sig)
    # prmt(d, a, a, sel) pattern: same register for both source operands
    prmt_self = []
    for si, st in enumerate(prog.steps):
        if st["op"] == "prmt.b32" and len(st["srcs"]) == 2 and st["srcs"][0] == st["srcs"][1]:
            prmt_self.append((si, st["srcs"][0]))
    # over-MSB bfe/bfi
    overmsb = []
    for si, st in enumerate(prog.steps):
        if st["op"] in ("bfe.u32", "bfe.s32", "bfe.u64", "bfe.s64", "bfi.b32", "bfi.b64") \
                and st["imm"] and st["imm"][0] + st["imm"][1] > (64 if st["op"].endswith("64") else 32):
            overmsb.append((si, st["imm"]))
    # set.* / bfind families
    set_ops = [op for op in present if op.startswith("set.")]
    bfind_ops = [op for op in present if op.startswith("bfind")]
    shf_ops = [op for op in present if op.startswith("shf.")]

    classes = []
    if prmt_self:
        classes.append(("prmt_self_operand", prmt_self[:4]))
    if overmsb:
        classes.append(("overmsb_bfe_bfi", overmsb[:4]))
    if set_ops:
        classes.append(("set_mixed_type", set_ops[:6]))
    if bfind_ops:
        classes.append(("bfind", bfind_ops[:4]))
    if shf_ops:
        classes.append(("shf", shf_ops[:4]))

    # compare with the known trigger classes
    known = {
        "prmt_self_operand": "ccf599c62b41dba5 (prmt(d,a,a,sel))",
        "overmsb_bfe_bfi": "ebf45243624dc5d6 (over-MSB bfe/bfi)",
        "set_mixed_type": "51e8b93f6d55b25a (set.* mixed-type)",
        "bfind": "51e8b93f6d55b25a (bfind.shiftamt)",
    }
    matches_known = [known[c[0]] for c in classes if c[0] in known]
    return {
        "classes": classes,
        "matches_known": matches_known if matches_known else (["NEW/UNCLASSIFIED"] if classes else []),
        "is_new_root_cause": bool(classes) and not any(c[0] in known for c in classes),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def ensure_report(run_dir: pathlib.Path) -> pathlib.Path:
    report = run_dir / "overnight_report.md"
    if not report.exists():
        report.write_text(
            "# Overnight cuda_stress run -- differential analysis\n\n"
            f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Each `potential_bug` case is audited for PTX-ISA UB and its root cause is\n"
            "classified against the known trigger classes "
            "(prmt(d,a,a,sel) / over-MSB bfe-bfi / set.*-mixed-type / bfind).\n\n---\n\n")
    return report


def analyze_case(case_dir: pathlib.Path, run_dir: pathlib.Path) -> dict:
    from pipeline.manifest import case_from_dir
    from pipeline import comparator
    from pipeline.config import load_config
    from generator.strategies.inline_ptx import _gen_program
    cfg = load_config()
    case = case_from_dir(case_dir)
    prog = _gen_program(case.kernel_seed, case.params)
    audit = ub_audit_program(prog)
    root = classify_root_cause(prog)
    result = comparator.compare_matrix(cfg, case, case_dir) if (case_dir / "builds").exists() else None
    return {
        "case_id": case.case_id(),
        "dir": str(case_dir.name),
        "strategy": case.strategy,
        "generator_version": case.generator_version,
        "status": result.status if result else "N/A",
        "ub_audit_problems": audit["problems"],
        "ub_audit_ok": not audit["problems"],
        "opcode_count": audit["opcode_count"],
        "steps": audit["steps"],
        "root_cause": root,
    }


def append_report_entry(report: pathlib.Path, entry: dict) -> None:
    lines = [
        f"## {entry['dir']}",
        f"- status      : {entry['status']}",
        f"- generator   : v{entry['generator_version']}  ({entry['strategy']})",
        f"- opcode count: {entry['opcode_count']} across {entry['steps']} steps",
        f"- UB audit    : {'CLEAN' if entry['ub_audit_ok'] else 'VIOLATIONS ' + json.dumps(entry['ub_audit_problems'])}",
        f"- root cause  : {json.dumps(entry['root_cause'], default=str)}",
        "",
    ]
    with open(report, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id, e.g. run-20260811-xxxx")
    ap.add_argument("--poll", type=float, default=120.0, help="poll interval seconds")
    ap.add_argument("--once", action="store_true", help="analyze all pending once and exit")
    args = ap.parse_args()

    run_dir = pathlib.Path("/home/linux/cuda_stress/runs") / args.run
    report = ensure_report(run_dir)
    analyzed = set()
    state = run_dir / ".overnight_analyzed.json"
    if state.exists():
        analyzed = set(json.loads(state.read_text()))

    while True:
        ledger = run_dir / "results.jsonl"
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                cid = row.get("case_id")
                if not cid or cid in analyzed:
                    continue
                if row.get("category") != "potential_bug":
                    analyzed.add(cid)
                    continue
                case_dir = run_dir / "tests" / cid
                if not (case_dir / "manifest.json").exists():
                    continue
                try:
                    entry = analyze_case(case_dir, run_dir)
                    append_report_entry(report, entry)
                    analyzed.add(cid)
                    print(f"[analyze] {cid}: UB={'CLEAN' if entry['ub_audit_ok'] else 'VIOLATION'} "
                          f"root={[c[0] for c in entry['root_cause']['classes']]}", flush=True)
                except Exception as e:
                    print(f"[analyze] {cid}: ERROR {e}", flush=True)
                    analyzed.add(cid)
        state.write_text(json.dumps(sorted(analyzed)))
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
