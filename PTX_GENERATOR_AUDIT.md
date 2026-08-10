# PTX ISA 9.3 generator audit

Normative source: `ptx_isa_reference/ptx_isa_9.3.{html,txt}`.

## Findings fixed

1. Plain `shl`/`shr` counts were over-constrained with `& 31`/`& 63`.
   PTX clamps counts greater than the register width to the width. The generator
   now sends unclipped u32 counts and covers `0`, `N-1`, `N`, `N+1`, and `255`.
2. The count source for 64-bit `shl`/`shr` was modeled as a 64-bit register.
   PTX requires operand `b` to be 32-bit regardless of destination width.
3. Generic `prmt` selectors were restricted to 0..7. Selector values 8..15 are
   valid and replicate the selected byte's sign bit.
4. `bfe`/`bfi` omitted zero-length, signed, 64-bit, position 255, and several
   cross-MSB combinations. These are now explicitly modeled.
5. Random selection did not guarantee that advertised rare operations appeared.
   `coverage_mode=balanced` now emits every eligible variant before repetition.
6. No PTX-version or target-SM gate existed. Every operation now carries
   `min_ptx` and `min_sm`; the default PTX 7.3 ceiling preserves compatibility
   with the configured CUDA 11.3 toolchain. PTX 7.6 operations such as `bmsk`
   and `szext` require an explicit higher ceiling.
7. Value-destroying boundary operations could make a generated case independent
   of runtime input. Final PTX-side input anchors now guarantee sensitivity.

## Coverage added

The inline strategy now contains 154 CPU-mirrored variants covering integer
arithmetic, signed/unsigned min/max, `mul24`/`mad24`, `mul.wide`, `mad.lo`,
nonzero-divisor `div`/`rem`, `abs`/`neg`, `bfind`/`fns`, 32/64-bit `bfe`/`bfi`,
`bmsk`/`szext` with ISA gates, full generic `prmt`, all four `shf` direction/mode
forms, clamped signed/unsigned shifts, `set`/`setp` comparisons, `selp`/`slct`,
and predicate/32/64-bit logic.

The manifest records `covered_ptx_variants` and `covered_ptx_mnemonics` for each
case. Balanced configuration variants include at least one case long enough to
traverse the complete PTX-7.3-compatible table.

## Coverage boundary

The repository is not comprehensive over the entire PTX ISA. Run:

```sh
python3 tools/audit_ptx_coverage.py --format markdown
```

The current structural audit reports:

- 154 inline PTX type/modifier variants;
- 46 of 131 distinct documented mnemonic roots deliberately covered (35.1%);
- 46 of 162 section-qualified instruction families covered (28.4%).

Section-qualified coverage is the meaningful number: integer `add`, for example,
does not count as floating-point or half-precision `add` coverage.

The largest remaining gaps are floating/half/mixed precision, vector video,
extended carry chains, memory atomics and memory-model scopes, async copy and
barriers, texture/surface operations, tensor cores, cluster/fabric operations,
and exceptional control-flow instructions. These need dedicated stateful or
floating-point oracles and architecture-specific launch/resource setup; treating
them as interchangeable scalar operations would create under-constrained tests.
