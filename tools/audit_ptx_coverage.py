#!/usr/bin/env python3
"""Compare repository generator coverage with NVIDIA PTX ISA instruction headings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEADINGS = ROOT / "ptx_isa_reference" / "headings.tsv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.strategies.inline_ptx import OPS, _mnemonic  # noqa: E402

# These strategies are written in CUDA C++ but deliberately cause the listed PTX
# instruction families to be emitted.  Keep the list explicit: SASS target-opcode
# strings such as LDG/ISETP are not a reliable PTX coverage declaration.
OTHER_STRATEGY_ROOTS = {
    "predicate_pressure": {"bra", "selp", "setp"},
    "integer_chain": {"bfe", "bfi", "brev", "clz", "lop3", "mul", "popc", "prmt", "shf"},
    "memory_address": {"ld", "st"},
    "warp_mixed": {"activemask", "match", "shfl", "vote"},
    "register_pressure": {"call", "ld", "st"},
}

# Section-qualified coverage avoids claiming that integer `add` also covers the
# floating, half, mixed-precision, and packed-vector `add` families.
SECTION_COVERAGE = {
    "9.7.1": {
        "abs", "add", "bfe", "bfi", "bfind", "bmsk", "brev", "clz", "div",
        "fns", "mad", "mad24", "max", "min", "mul", "mul24", "neg", "popc",
        "rem", "sad", "sub", "szext",
    },
    "9.7.6": {"selp", "set", "setp", "slct"},
    "9.7.8": {"and", "cnot", "lop3", "not", "or", "shf", "shl", "shr", "xor"},
    "9.7.9": {"cvt", "ld", "mov", "prmt", "shfl", "st"},
    "9.7.13": {"bra", "call"},
    "9.7.14": {"activemask", "match", "vote"},
}


def documented_families(headings: Path) -> tuple[set[str], dict[str, set[str]]]:
    roots: set[str] = set()
    sections: dict[str, set[str]] = defaultdict(set)
    for line in headings.read_text(encoding="utf-8").splitlines()[1:]:
        columns = line.split("\t", 2)
        if len(columns) != 3:
            continue
        title = columns[2]
        match = re.match(r"(9\.7\.\d+).*Instructions:\s*(.*)", title)
        if not match:
            continue
        section, names = match.groups()
        names = re.sub(r"\([^)]*\)", "", names).replace("/", ",")
        for item in names.split(","):
            token_match = re.search(r"(?<![A-Za-z])[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*", item)
            if not token_match:
                continue
            root = token_match.group(0).split(".", 1)[0]
            if root in {"asynchronous", "bulk", "copy", "tensor", "non", "miscellaneous"}:
                continue
            roots.add(root)
            sections[section].add(root)
    return roots, dict(sections)


def coverage_report(headings: Path) -> dict:
    documented, sections = documented_families(headings)
    inline_variants = {op["name"] for op in OPS}
    inline_roots = {_mnemonic(name) for name in inline_variants}
    by_strategy = {name: sorted(values) for name, values in OTHER_STRATEGY_ROOTS.items()}
    by_strategy["inline_ptx"] = sorted(inline_roots)
    covered = set().union(*(set(values) for values in by_strategy.values()))
    documented_section_pairs = {(section, name) for section, names in sections.items() for name in names}
    covered_section_pairs = {
        (section, name)
        for section, names in SECTION_COVERAGE.items()
        for name in names
        if (section, name) in documented_section_pairs
    }
    return {
        "source": str(headings),
        "documented_instruction_families": len(documented),
        "covered_instruction_families": len(documented & covered),
        "coverage_percent": round(100.0 * len(documented & covered) / len(documented), 1),
        "documented_section_families": len(documented_section_pairs),
        "covered_section_families": len(covered_section_pairs),
        "section_coverage_percent": round(
            100.0 * len(covered_section_pairs) / len(documented_section_pairs), 1
        ),
        "inline_ptx_variants": len(inline_variants),
        "covered": sorted(documented & covered),
        "missing": sorted(documented - covered),
        "generator_only_names": sorted(covered - documented),
        "by_strategy": by_strategy,
        "missing_by_section": {
            section: sorted(names - SECTION_COVERAGE.get(section, set()))
            for section, names in sorted(sections.items())
            if names - SECTION_COVERAGE.get(section, set())
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# PTX ISA 9.3 generator coverage audit",
        "",
        f"- Documented instruction families: {report['documented_instruction_families']}",
        f"- Covered instruction families: {report['covered_instruction_families']}",
        f"- Family coverage: {report['coverage_percent']}%",
        f"- Section-qualified families: {report['covered_section_families']} / "
        f"{report['documented_section_families']} ({report['section_coverage_percent']}%)",
        f"- Inline PTX modeled variants: {report['inline_ptx_variants']}",
        "",
        "Coverage here means that at least one generator strategy deliberately emits the",
        "instruction family. It does not claim full modifier/type/state-space coverage.",
        "",
        "## Covered families",
        "",
        ", ".join(f"`{name}`" for name in report["covered"]),
        "",
        "## Missing families",
        "",
        ", ".join(f"`{name}`" for name in report["missing"]),
        "",
        "## Missing by ISA section",
        "",
    ]
    for section, names in report["missing_by_section"].items():
        lines.append(f"- `{section}`: " + ", ".join(f"`{name}`" for name in names))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headings", type=Path, default=DEFAULT_HEADINGS)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    report = coverage_report(args.headings)
    print(json.dumps(report, indent=2, sort_keys=True) if args.format == "json" else markdown(report), end="")


if __name__ == "__main__":
    main()
