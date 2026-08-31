#!/usr/bin/env python3
"""
Format inline PTX assembly: split \n\t separators so each instruction
occupies its own line as a complete C string literal (using implicit
string concatenation).  Idempotent — safe to run on already-formatted files.

Usage:
  python3 fmt_ptx.py file.cu           # in-place
  python3 fmt_ptx.py file.cu --stdout  # print to stdout
  python3 fmt_ptx.py file1.cu file2.cu # multiple files
"""

import re
import sys
import os


def format_file(path, in_place=True):
    with open(path, 'r') as f:
        text = f.read()

    # Match asm volatile(...) blocks.
    # After every asm block:
    #   Each instruction line is  "instr;\n\t"  (or  "instr;"  for the last).
    #   Adjacent C string literals are concatenated by the compiler.

    result = []
    pos = 0

    while True:
        m = re.search(r'\basm\s+volatile\s*\(', text[pos:])
        if not m:
            result.append(text[pos:])
            break

        # start / end of "asm volatile(" in the full text
        asm_start = pos + m.start()
        paren_open = pos + m.end() - 1          # position of '('
        result.append(text[pos:asm_start])       # everything before the asm keyword

        # --- walk parentheses to find the matching ) ---
        i = paren_open + 1
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            i += 1
        # i is now one past the closing )

        # --- determine base indentation ---
        line_start = text.rfind('\n', 0, asm_start)
        line_start = line_start + 1 if line_start != -1 else 0
        base_indent = re.match(r'^([ \t]*)', text[line_start:]).group(1)
        instr_indent = base_indent + '    '

        # asm_body = content between '(' and ')'
        asm_body = text[paren_open + 1:i - 1]

        # --- extract the asm template string ---
        # It is the first "…" block before the first output-operand colon (:"=)
        tmpl_start = re.search(r'"', asm_body)
        if not tmpl_start:
            result.append(text[asm_start:i])
            pos = i
            continue

        after_open = asm_body[tmpl_start.end():]

        # The closing " is right before  :  "=
        tmpl_end = re.search(r'"\s*:\s*"=', after_open)
        if not tmpl_end:
            result.append(text[asm_start:i])
            pos = i
            continue

        template_body = after_open[:tmpl_end.start()]
        suffix = after_open[tmpl_end.start():]   # e.g.  "\n        : \"=l\"(o0)…"

        # Remove any trailing newline from suffix (it belongs to the asm statement,
        # not to this instruction line — we emit a fresh newline below).
        suffix = suffix.rstrip('\n')

        # --- split template on literal \n\t ---
        # This works regardless of whether the source was one long line or already split.
        raw_instructions = template_body.split('\\n\\t')
        # Clean: strip whitespace, then remove any embedded quotes/newlines/tabs
        # (handles both original one-line and already-formatted idempotent re-runs)
        instructions = []
        for ins in raw_instructions:
            clean = ins.strip().replace('"', '').replace('\n', '').replace('\t', '').strip()
            if clean:
                instructions.append(clean)

        # --- rebuild ---
        result.append(f'asm volatile(\n')

        if instructions:
            for idx, ins in enumerate(instructions):
                if idx == len(instructions) - 1:
                    # Last instruction — suffix already provides the closing "
                    result.append(f'{instr_indent}"{ins}{suffix}\n')
                else:
                    result.append(f'{instr_indent}"{ins}\\n\\t"\n')

        # closing ) and optional ;
        trailer = ')'
        if i < len(text) and text[i] == ';':
            trailer += ';'
            i += 1
        result.append(trailer)
        pos = i

    output = ''.join(result)

    if in_place:
        with open(path, 'w') as f:
            f.write(output)
        print(f'Formatted: {path}')
    else:
        sys.stdout.write(output)


def main():
    args = sys.argv[1:]
    to_stdout = False
    if '--stdout' in args:
        to_stdout = True
        args.remove('--stdout')

    if not args:
        print(__doc__)
        sys.exit(1)

    for path in args:
        if not os.path.isfile(path):
            print(f'Skipping (not a file): {path}', file=sys.stderr)
            continue
        format_file(path, in_place=not to_stdout)


if __name__ == '__main__':
    main()
