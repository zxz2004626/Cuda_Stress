#!/usr/bin/env python3
"""Extract searchable text and a heading index from NVIDIA's single-page PTX ISA HTML."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup


def normalize_text(raw: str) -> str:
    raw = raw.replace("", "").replace("\r\n", "\n")
    lines = [line.rstrip() for line in raw.split("\n")]
    output: list[str] = []
    blank = False
    for line in lines:
        if line.strip():
            output.append(line)
            blank = False
        elif not blank:
            output.append("")
            blank = True
    return "\n".join(output).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("text", type=Path)
    parser.add_argument("headings", type=Path)
    args = parser.parse_args()

    soup = BeautifulSoup(args.html.read_bytes(), "html.parser")
    document = soup.select_one('div[role="main"].document')
    if document is None:
        raise SystemExit("PTX document body was not found")

    for node in document.select("script, style, nav"):
        node.decompose()

    # Sphinx already places block-level whitespace in the source. Adding a
    # separator here would split inline elements (links, code spans, emphasis)
    # onto separate lines and make instruction constraints harder to search.
    args.text.write_text(normalize_text(document.get_text()), encoding="utf-8")

    rows = ["level\tanchor\ttitle"]
    for heading in document.find_all(re.compile(r"^h[1-6]$")):
        level = heading.name[1:]
        anchor = heading.get("id", "")
        if not anchor and heading.parent is not None:
            anchor = heading.parent.get("id", "")
        title = " ".join(heading.stripped_strings).replace("\t", " ").replace("", "")
        rows.append(f"{level}\t{anchor}\t{title}")
    args.headings.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
