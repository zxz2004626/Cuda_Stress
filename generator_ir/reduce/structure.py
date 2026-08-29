"""Structural hashing and size metrics for generator_ir Programs.

IR nodes are ``@dataclass(eq=False)`` with ``hash=id``.  Caching and logging
therefore need a content-derived digest and size counters that never use
``id(self)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from generator_ir.ir.typedecl import GType
from generator_ir.ir.program import Program, LaunchConfig
from generator_ir.ir.statements import RawCode
from generator_ir.ir.traversal import traverse


def type_key(t: Any) -> str:
    """Stable string for GType objects (repr contains no object id)."""
    return repr(t)


def _enum_key(e: Enum) -> str:
    return f"{type(e).__name__}.{e.name}"


def _ser(x: Any, depth: int = 0) -> str:
    """Deterministic one-line serialization of a value tree.

    The IR is a tree for all practical purposes (no parent pointers).  If a
    node were accidentally shared this simply serializes it twice, which is
    still deterministic.
    """
    if x is None:
        return "N"
    if isinstance(x, GType):
        return f"T({type_key(x)})"
    if isinstance(x, Enum):
        return f"E({_enum_key(x)})"
    if isinstance(x, (bool, int, float)):
        return repr(x)
    if isinstance(x, str):
        return f"S({x!r})"
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(_ser(v, depth + 1) for v in x) + "]"
    if isinstance(x, (set, frozenset)):
        return "{" + ",".join(_ser(v, depth + 1) for v in sorted(x, key=repr)) + "}"
    if is_dataclass(x):
        parts = []
        for f in fields(x):
            if f.name == "emit_ctx":
                continue
            parts.append(f"{f.name}={_ser(getattr(x, f.name), depth + 1)}")
        return f"{type(x).__name__}({','.join(parts)})"
    return f"O({type(x).__name__}:{x!r})"


def _serialize_program(program: Program) -> str:
    parts = ["Program"]
    if isinstance(program.launch, LaunchConfig):
        parts.append(_ser(program.launch))
    else:
        parts.append(_ser(program.launch))
    parts.append("decls=[" + ",".join(_ser(d) for d in program.decls) + "]")
    return "\n".join(parts)


def structural_digest(program: Program) -> str:
    """Content-derived sha256 of the whole IR tree."""
    text = _serialize_program(program)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def program_counts(program: Program) -> dict:
    """Size metrics used by reduce_log.jsonl and report.json."""
    nodes = list(traverse(program))
    stmt_types = (
        "Assign", "CompoundAssign", "IncDecStmt", "ExprStmt", "Declare",
        "Block", "If", "For", "While", "DoWhile", "Switch", "Case",
        "Return", "Break", "Continue", "Barrier", "Fence", "UnrollHint",
        "RawCode",
    )
    stmt_count = sum(1 for n in nodes if type(n).__name__ in stmt_types)
    expr_count = sum(1 for n in nodes if type(n).__name__ not in stmt_types
                     and not type(n).__name__ in {"Program", "Func", "StructDecl"})
    rawcode_count = sum(1 for n in nodes if isinstance(n, RawCode))
    return {
        "stmt_count": stmt_count,
        "expr_count": expr_count,
        "rawcode_count": rawcode_count,
        "node_count": len(nodes),
    }
