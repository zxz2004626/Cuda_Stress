# NVIDIA PTX ISA reference

- Source: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Saved: 2026-08-10
- Document version: PTX ISA 9.3

Files:

- `ptx_isa_9.3.html`: unmodified single-page HTML downloaded from NVIDIA.
- `ptx_isa_9.3.txt`: searchable UTF-8 text extracted from the document body.
- `headings.tsv`: heading level, HTML anchor, and title for section lookup.
- `extract_text.py`: reproducible HTML-to-text/index extractor.

Regenerate the derived files with:

```sh
python3 extract_text.py ptx_isa_9.3.html ptx_isa_9.3.txt headings.tsv
```

For instruction generation, search `headings.tsv` for the instruction name and
then consult the corresponding section in the HTML or text. In particular,
check Syntax, Semantics, Notes, PTX ISA Notes, and Target ISA Notes together;
an instruction's valid types, modifiers, operand ranges, required PTX version,
and minimum target architecture can be distributed across those subsections.

The downloaded document contains NVIDIA's notices and terms in Chapter 14.
