"""Struct / aggregate weaving block (port of perffuzzer gens_struct).

Exercises the C++ aggregate front-end path -- local struct definition, member
writes/reads, and whole-struct copy -- which no template or other block emits.
This stresses struct layout, member-access codegen, and aggregate copy in the
two compilers.

Determinism: every field is a deterministic function of pool values + tid, the
copy is bitwise, and all arithmetic is unsigned (defined overflow) -- so the
folded value is stable across runs and across compiler versions.  No sync /
shared / warp state, so it is context-free (not requires_uniform) and portable
to HIP.

NOTE: this is one of the 12 feature-coverage blocks -- registered with
``weight=0.0`` (disabled by default), mirroring perffuzzer's
``_FEATURE_COVERAGE_GENS`` mechanism where ``struct_aggregate`` gets weight 0
unless feature gens are explicitly enabled.
"""

from __future__ import annotations

from generator_ir.ir.typedecl import T
from generator_ir.ir.nodes import Expr, Var, Lit, Cast
from generator_ir.ir.statements import RawCode
from generator_ir.gen.weave_handlers import weave_block


def _c_str(e: Expr) -> str:
    """Render a pool-read expr as C source for the raw block below.

    perffuzzer relied on ``Expr.__str__`` producing C text; gpuperf nodes have
    no ``__str__``, so the small subset ``ctx.kx_random_var`` can return (Var,
    Cast, or a Lit fallback) is rendered here."""
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Cast):
        return f"({e.to.spelling()})({_c_str(e.arg)})"
    if isinstance(e, Lit):
        return str(e.value)
    raise TypeError(f"cannot render {type(e).__name__} as C source")


@weave_block("struct_aggregate", inputs=(T.u32,), weight=0.0, cost=2)
def _blk_struct_aggregate(ctx) -> None:
    sname = ctx.kx_fresh_name("Agg")
    sv = ctx.kx_fresh_name("agg")
    sv2 = ctx.kx_fresh_name("agg")
    a = _c_str(ctx.kx_random_var(T.u32, recency=0.6))
    b = _c_str(ctx.kx_random_var(T.u32, recency=0.5))
    code = (
        "{ // local struct aggregate (definition, member access, copy)\n"
        f"    struct {sname} {{ unsigned int a; unsigned int b; unsigned int c; }};\n"
        f"    {sname} {sv};\n"
        f"    {sv}.a = (unsigned int)({a});\n"
        f"    {sv}.b = ((unsigned int)({b})) ^ ((unsigned int)(tid));\n"
        f"    {sv}.c = ({sv}.a * 2654435761u) + ({sv}.b ^ 0x9E3779B9u);\n"
        f"    {sname} {sv2} = {sv};\n"
        f"    _fv = (((_fv) * 2654435761u) ^ {sv2}.a ^ ({sv2}.b << 1) ^ {sv2}.c);\n"
        "}"
    )
    ctx.emit(RawCode(text=code))
