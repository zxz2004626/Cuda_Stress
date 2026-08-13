"""Classic-kernel-template fragment adapters (ported from perffuzzer
``generator/codegen/gens_template.py``).

Each gpuperf classic template (``gpuperf/templates/``) already ships an
*observable_fold* "flat" form: in weave mode (``ctx.config.observable_fold``
True, ``ctx.kx_is_2d`` False) ``build_body(ctx)`` returns a FLAT chunk of
statements instead of owning the whole kernel -- it declares its scratch into
the shared variable pool and folds its per-thread result into a SAMPLED
in-kx_scope sink (via ``ctx.kx_sample_or_create(for_write=True)``).  That is exactly
the shape a fragment weave block needs.

So this module is a thin adapter: for every template that can act as a
self-contained fragment we register a ``@weave_block`` that

  1. fetches the template from the default registry,
  2. runs its flat body and ``ctx.emit``s each returned statement into the
     CURRENT (uniform, function-kx_scope) block,
  3. recovers the fragment's result sink(s) -- both the vars the template chose
     via ``kx_sample_or_create(for_write=True)`` AND every sink-eligible integer
     var it newly declared at function kx_scope -- and folds them into ``_fv``
     via ``ctx.observe(...)`` (otherwise the fragment is DCE'd and contributes
     nothing to the checksum),
  4. declares ``requires_uniform=True`` for every template that emits
     ``__syncthreads``/shared-memory/warp collectives (so it only fires at the
     converged top level -> race-/deadlock-free, deterministic across
     schedulings), and a ``cost`` proportional to the fragment's size.

SKIPPED templates (they fundamentally own kernel structure -> whole-kernel
WeaveTemplate tier, NOT a fragment):
  * tiled_gemm  -- ``kx_is_2d()`` True (2D y*n+x output indexing).
  * tile_swap   -- ``kx_is_2d()`` True (2D tiled output indexing).
  * prefix_scan -- ``uses_non_tid_output()`` True (writes output[idx],
                   idx != tid; its flat form would still need exclusive
                   ownership of the output layout).
"""

from __future__ import annotations

from typing import List, Set

from generator_ir.ir.typedecl import (
    GType, T, TBool, TChar, TSChar, TShort, TUShort, TInt, TUInt, TLong,
    TULong, TArray, TPointer, TQualified,
)
from generator_ir.ir.nodes import Var, Expr
from generator_ir.gen.context import GenContext
from generator_ir.gen.weave_handlers import weave_block


# Integer (non-pointer, non-array) scalar types that are safe to fold into the
# deterministic XOR checksum via ctx.observe().
_OBSERVE_INT_TYPES = (
    TUInt, TInt, TULong, TLong,
    TUShort, TShort, TChar, TSChar, TBool,
)

# Names that must never be re-observed (params / running-fold / loop vars).
_OBSERVE_RESERVED = frozenset({
    "tid", "gid", "n", "output", "input", "_fv", "_rep", "_ci",
    "output_fp_acc",
})


def _kx_unqual(t: GType) -> GType:
    return t.unqualified() if isinstance(t, TQualified) else t


def _kx_is_observable_int(var: Var) -> bool:
    """True if *var* is a scalar integer var safe to fold into _fv."""
    if var.name in _OBSERVE_RESERVED or var.name.startswith("i_"):
        return False
    vt = _kx_unqual(var.ty)
    if isinstance(vt, (TPointer, TArray)):
        return False
    return isinstance(vt, _OBSERVE_INT_TYPES)


def _fragment_registry(ctx: GenContext):
    """The cached classic-template registry (shared with stmt_builder's
    template_fragment statement kind)."""
    reg = getattr(ctx, "_fragment_registry", None)
    if reg is None:
        from generator_ir.templates.catalog import kx_build_default_registry
        reg = kx_build_default_registry()
        ctx._fragment_registry = reg
    return reg


def _make_template_block(registry_name: str, *,
                         requires_uniform: bool, cost: int):
    """Build + register a fragment weave block that runs the classic template
    *registry_name* in flat mode, emits its statements, and observes its
    result sink(s) into _fv."""

    # Template fragments fire at most ONCE per kernel: several wrap legacy
    # templates emitting hardcoded verbatim names (e.g. perm_round's _pc_*)
    # that collide if the same fragment is woven twice in one kx_scope.  One big
    # template block per kernel is also the intended granularity.
    @weave_block("tmpl_" + registry_name, inputs=(T.u32,),
                 requires_uniform=requires_uniform, cost=cost, once=True)
    def _blk(ctx: GenContext, _name: str = registry_name) -> None:
        template = _fragment_registry(ctx).get(_name)

        # Only 1D tid-output templates can act as pool-sharing fragments, and
        # only in a non-2D kernel (in 2D mode ``kx_flat_mode`` is off, so
        # build_body would return the whole-kernel form).  Also guard against
        # recursion from a nested fragment emission path.
        if (template is None or template.kx_is_2d()
                or template.uses_non_tid_output()
                or getattr(ctx, "_in_template_fragment", False)
                or ctx.kx_is_2d):
            return

        # Snapshot the function-kx_scope integer vars that exist BEFORE the
        # template runs, so afterward we can fold in every sink-eligible var
        # the template NEWLY declared (its scratch results) -- these are
        # guaranteed reachable.
        before: Set[str] = {
            v.name for v in ctx.kx_function_scope_variables()
            if _kx_is_observable_int(v)
        }

        # Record the writable sinks the template chooses via kx_sample_or_create:
        # the template folds its per-thread RESULT into one of these (which
        # may be a PRE-EXISTING pool var, missed by the new-var delta above).
        # We wrap kx_sample_or_create for the duration of this fragment to
        # capture them.
        chosen_sinks: List[Var] = []
        orig_sample = ctx.kx_sample_or_create

        def _kx_tracking_sample(gtype, for_write=False, init=None):
            var, decl = orig_sample(gtype, for_write=for_write, init=init)
            if for_write and isinstance(var, Var):
                chosen_sinks.append(var)
            return var, decl

        ctx.kx_sample_or_create = _kx_tracking_sample
        ctx._in_template_fragment = True
        try:
            # observable_fold is on (weave mode) and ctx.kx_is_2d is False, so
            # build_body returns the FLAT statement list.
            stmts = template.build_body(ctx)
        finally:
            ctx._in_template_fragment = False
            ctx.kx_sample_or_create = orig_sample

        for stmt in stmts:
            ctx.emit(stmt)

        # Fold the fragment's result into _fv.  Union of:
        #   * the sinks the template wrote its result into (kx_sample_or_create
        #     for_write -- the canonical result carriers), and
        #   * every sink-eligible integer var newly declared at function kx_scope
        #     (covers fresh "v_*" sinks + any folded scratch results).
        observed: Set[str] = set()
        for sink in chosen_sinks:
            if _kx_is_observable_int(sink) and sink.name not in observed:
                ctx.observe(Var(_kx_unqual(sink.ty), name=sink.name))
                observed.add(sink.name)

        for v in ctx.kx_function_scope_variables():
            if (v.name not in before and v.name not in observed
                    and _kx_is_observable_int(v)):
                ctx.observe(Var(_kx_unqual(v.ty), name=v.name))
                observed.add(v.name)

    return _blk


# ---------------------------------------------------------------------------
# Registrations.
#
# requires_uniform=True for every template that emits __syncthreads / shared
# memory / warp collectives in its flat body (so it fires only at the
# converged top level -> race-/deadlock-free + scheduling-independent).
# map_transform is ALSO uniform because its flat body weaves stmt-gen output,
# which may emit a __syncthreads().  isa_stress is purely thread-local (no
# shared, no barriers), so it is safe in any context -> requires_uniform=False.
#
# cost is proportional to the fragment's emitted size (number of structural
# statements / pattern groups), so the weaving budget spends fairly across the
# small (map_transform) and large (vec_load_probe / isa_stress) fragments.
#
# Names mirror the gpuperf classic-template registry:
#   elementwise->map_transform, reduction->tree_reduce, stencil->neighbor_sum,
#   bank_conflict->smem_bank_stress, switch_divergence->cf_switch_tree,
#   warp_comm_fallback->lane_exchange, barrier_pressure->sync_span_live,
#   gather_scatter->irregular_io, histogram->bin_count,
#   register_pressure->reg_spill_probe, vectorization_stress->vec_load_probe,
#   permutation_comm->perm_round, ptx_stress->isa_stress.
# ---------------------------------------------------------------------------

_make_template_block("map_transform", requires_uniform=True, cost=3)
_make_template_block("tree_reduce", requires_uniform=True, cost=6)
_make_template_block("neighbor_sum", requires_uniform=True, cost=5)
_make_template_block("smem_bank_stress", requires_uniform=True, cost=6)
_make_template_block("cf_switch_tree", requires_uniform=True, cost=5)
_make_template_block("lane_exchange", requires_uniform=True, cost=5)
_make_template_block("sync_span_live", requires_uniform=True, cost=5)
_make_template_block("irregular_io", requires_uniform=True, cost=5)
_make_template_block("bin_count", requires_uniform=True, cost=6)
_make_template_block("reg_spill_probe", requires_uniform=True, cost=6)
_make_template_block("vec_load_probe", requires_uniform=True, cost=7)
_make_template_block("perm_round", requires_uniform=True, cost=6)
_make_template_block("isa_stress", requires_uniform=False, cost=6)
