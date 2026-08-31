"""PTX / SASS parsing: instruction counts, opcode histograms, normalized hashes.

`normalized_hash` is robust to register allocation (registers/addresses are canonicalized)
but sensitive to instruction selection, scheduling and optimization, which is exactly what
we want for differential SASS comparison across ptxas profiles.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List

# cuobjdump SASS lines end with "; <trailing hex comment>".  Capture the operands up to
# the terminating ';' and ignore the trailing /* 0x.... */ comment.
SASS_INST_RE = re.compile(
    r'^\s*/\*[0-9a-f]+\*/\s*(?:@!?P\d+\s+)?([A-Za-z0-9][A-Za-z0-9._]*)(?:\s+(.*?))?;\s*(?:/\*.*)?$')

SASS_OPCODE_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._]*')


def _sass_normalize_line(opcode: str, operands: str) -> str:
    ops = operands or ""
    ops = re.sub(r'RZ', 'RZ', ops)
    ops = re.sub(r'\bR\d+\b', 'R', ops)
    ops = re.sub(r'\bUR\d+\b', 'U', ops)
    ops = re.sub(r'\bP\d+\b', 'P', ops)
    ops = re.sub(r'c\[0x[0-9a-f]+\]\[0x[0-9a-f]+\]', 'C', ops)
    ops = re.sub(r'0x[0-9a-fA-F]+', 'I', ops)
    ops = re.sub(r'-?\d+', 'I', ops)
    ops = re.sub(r'L\d+', 'L', ops)
    ops = re.sub(r'\s+', ' ', ops).strip()
    return f"{opcode} {ops}"


def parse_sass(text: str) -> Dict[str, Any]:
    """Parse cuobjdump -sass output.

    normalized_hash canonicalizes registers/addresses so register-allocation differences
    do not create false positives, while keeping instruction selection, ordering and
    operand structure (which ptxas -O levels genuinely change).
    """
    insts: List[str] = []          # opcode only (order-sensitive, alloc-insensitive)
    norm: List[str] = []           # opcode + canonicalized operands
    hist: Dict[str, int] = {}
    branches = predicated = loads = stores = 0
    for line in text.splitlines():
        m = SASS_INST_RE.match(line)
        if not m:
            continue
        opcode, operands = m.group(1), m.group(2) or ""
        insts.append(opcode)
        norm.append(_sass_normalize_line(opcode, operands))
        base = opcode.split('.')[0]
        hist[base] = hist.get(base, 0) + 1
        if opcode.startswith(("BRA", "BSSY", "BSYNC", "BSSB", "BRX", "BSP")):
            branches += 1
        if "@" in line[:40]:
            predicated += 1
        if opcode.startswith(("LDG", "LD", "LDL", "LDS", "LDC")):
            loads += 1
        if opcode.startswith(("STG", "ST", "STL", "STS", "STC")):
            stores += 1
    return {
        "instruction_count": len(insts),
        "opcode_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "normalized_hash": hashlib.sha256("\n".join(norm).encode()).hexdigest(),
        "opcode_order_hash": hashlib.sha256("\n".join(insts).encode()).hexdigest(),
        "branch_count": branches,
        "predicated_count": predicated,
        "load_count": loads,
        "store_count": stores,
    }


# PTX: skip directives/declarations; an instruction is a non-directive, non-label line.
PTX_INST_RE = re.compile(
    r'^\s*(?:@!?P\d+\s+)?([a-z][a-z0-9._]*)\s+', re.IGNORECASE)
PTX_SKIP_RE = re.compile(
    r'^\s*(?:\.\w+|//|@\w|#|\{|\}|\/\*|bra\s|//)')


def parse_ptx(text: str) -> Dict[str, Any]:
    hist: Dict[str, int] = {}
    insts: List[str] = []
    predicated = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("//") or s.startswith((".", "}")):
            continue
        if s.startswith("{") or s.startswith("}"):
            continue
        m = PTX_INST_RE.match(s)
        if not m:
            continue
        op = m.group(1)
        insts.append(op)
        base = op.split('.')[0]
        hist[base] = hist.get(base, 0) + 1
        if s.startswith("@"):
            predicated += 1
    norm_hash = hashlib.sha256("\n".join(insts).encode()).hexdigest()
    return {
        "instruction_count": len(insts),
        "opcode_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "normalized_hash": norm_hash,
        "predicated_count": predicated,
    }


def parse_ptxas_verbose(text: str) -> Dict[str, Any]:
    m = re.search(r'Used (\d+) registers', text)
    regs = int(m.group(1)) if m else None
    stack = spill_s = spill_l = None
    m = re.search(r'(\d+) bytes stack frame', text)
    if m:
        stack = int(m.group(1))
    m = re.search(r'(\d+) bytes spill stores', text)
    if m:
        spill_s = int(m.group(1))
    m = re.search(r'(\d+) bytes spill loads', text)
    if m:
        spill_l = int(m.group(1))
    return {"registers": regs, "stack_frame_bytes": stack,
            "spill_stores_bytes": spill_s, "spill_loads_bytes": spill_l}


def sass_target_opcode_count(metrics: Dict[str, Any], target_opcodes: List[str]) -> int:
    """Count SASS instructions whose base opcode matches any target opcode prefix."""
    hist = metrics.get("opcode_histogram", {})
    total = 0
    for op, count in hist.items():
        if any(op.startswith(t) for t in target_opcodes):
            total += count
    return total
