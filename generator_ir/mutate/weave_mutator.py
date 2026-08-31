"""Weave-mode splice mutator (port of perffuzzer's CodeGenMutator).

A Fuzzilli-style insertion mutator: it reuses the two-tier weaving builder
(``GenContext`` + the ``WeaveBlock`` registry) to INSERT freshly woven,
_fv-observed code into an EXISTING weave-shaped kernel, preferring a splice
inside the amplified ``_rep`` repeat loop so produced values fold into the
SAME running checksum ``_fv`` and feed the loop-carried state.

Scope:
  * Applies ONLY to weave-shaped kernels (``kernel_has_fv_write`` -- the 1D
    ``output[tid] = _fv`` OR the 2D ``output[idx] ^= _fv`` shape).  A
    non-weave kernel is returned unchanged.
  * Picks a RECORDED-UNIFORM insertion point (usually in the repeat-loop
    body, falling back to the kernel top level) using the ``Stmt.emit_ctx``
    snapshot stamped by the builder, so the reconstructed context is marked
    uniform there and ``requires_uniform`` (sync/shared) weave blocks may
    fire; at a non-uniform point only geometry-agnostic blocks run.
  * Reconstructs a ``GenContext`` whose variable pool reflects the kernel's
    in-kx_scope integer vars at that point, with ``_fv`` already declared
    (``_fv_declared=True``), ``_weave_mode``/``observable_fold`` on, and
    ``_block_is_1d`` set per the kernel shape.
  * Runs a small ``ctx.weave`` budget; the woven blocks ``ctx.observe`` their
    produced values into the existing ``_fv``.  The new statements are
    spliced before the loop-carried ``_lc_*`` tail when that tail is present.

Invariants (so the observable checksum semantics survive mutation):
  * Deterministic + UB-free by construction (the weave blocks are; the splice
    does not reorder or duplicate the single output write, nor append work
    after the checksum write).
  * Honors ``allow_volatile`` (passed through ``GenConfig``).
  * Newly declared variable names are made unique against the host kernel so
    the splice never shadows / redeclares an existing name.

Registration: the orchestrator adds this mutator to the pool ONLY when the
pipeline is constructed with ``codegen=True``.
"""

from __future__ import annotations

import contextlib
import random
import re
from typing import List, Optional

from generator_ir.ir.typedecl import (
    GType, T, TArray, TPointer, TQualified,
    TUInt, TInt, TULong, TLong, TUShort, TShort, TChar, TSChar, TBool,
)
from generator_ir.ir.operators import Arith
from generator_ir.ir.nodes import Expr, Var, BinOp, Cast
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, If, For, Block, RawCode,
)
from generator_ir.ir.program import Func, Program
from generator_ir.ir.traversal import (
    clone_program, kernel_of, collect, declarations, all_var_names,
    unqualified,
)
from generator_ir.ir.safety_checks import kernel_has_fv_write, _is_fv_output_write
from generator_ir.mutate.foundation import Mutation

# Matches a generator-style ``<prefix>_<number>`` name (e.g. cv_3, wsmem_0).
_NAME_RE = re.compile(r"^(.*)_(\d+)$")

# Matches a C scalar/array DECLARATION head inside RawCode text:
#   [__shared__|static] [volatile] [const] [unsigned] <base-type> NAME
# capturing NAME.  References / assignments (no leading type) are NOT matched.
_VERBATIM_DECL_RE = re.compile(
    r"\b(?:__shared__\s+|static\s+|volatile\s+|const\s+)*"
    r"(?:unsigned\s+|signed\s+)?"
    r"(?:long\s+long|long|int|short|char|float|double|__half|__nv_bfloat16)"
    r"\s+([A-Za-z_]\w*)")

# Unambiguous fresh-pool variable references (dangling-ref safety net).
_POOL_REF_RE = re.compile(r"\b(?:cv|v|ov)_\d+\b")

# Integer (non-pointer) types the weave pool can sample (mirrors
# GenContext.POOL_TYPES).
_LOOSE_INT_TYPES = (
    TUInt, TInt, TULong, TLong, TUShort, TShort, TChar, TSChar, TBool,
)


@contextlib.contextmanager
def _insert_blocks_disabled():
    """Temporarily remove weave blocks unsafe/noisy for splice mutation.

    FP blocks need extra host plumbing (``output_fp``) that a splice cannot
    add to an already-built kernel.  Template-fragment blocks (``tmpl_*``)
    often contain hardcoded scratch names; the base kernel already has
    template fragments, so a second fragment in the same kx_scope frequently
    collides.  A mutation can still interact with template code by reading
    its existing in-kx_scope variables from the repeat-loop body.

    Restores the registry to its EXACT original contents AND ORDER on exit:
    ``pick_block`` draws via ``rng.choices(pool, ...)`` where ``pool`` is in
    REGISTRY iteration order, so a reordered registry would change selection
    for the same RNG state -> cross-call nondeterminism.  We rebuild the dict
    in-place from a snapshot to preserve order."""
    from generator_ir.gen import weave_handlers as _wh
    snapshot = dict(_wh.REGISTRY)  # preserves insertion order
    for n in [k for k in _wh.REGISTRY
              if k.startswith("fp_") or k.startswith("tmpl_")]:
        del _wh.REGISTRY[n]
    try:
        yield
    finally:
        _wh.REGISTRY.clear()
        _wh.REGISTRY.update(snapshot)


class WeaveInsertMutator(Mutation):
    """Insert freshly woven, observed code into a weave-shaped kernel."""

    name = "CODEGEN_insert"
    # Bucketed with the compute family so category-based pool helpers keep
    # working; it is only ever added under the codegen gate.
    category = "compute"

    def perturb(self, program: Program, rng: random.Random, param=None) -> Program:
        prog = clone_program(program)
        kernel = kernel_of(prog)
        if kernel is None or kernel.body is None:
            return prog
        # Operate only on weave-shaped kernels (ignore everything else).
        if not kernel_has_fv_write(kernel):
            return prog

        # 1) Locate the trailing output-write If (the single checksum write).
        #    Prefer inserting into the uniform `_rep` loop body: that is where
        #    the woven statement kx_scope lives after amplification, so a splice
        #    can read and mix the same `cv_*`/template locals instead of
        #    living in an isolated top-level kx_scope.
        stmts = kernel.body.stmts
        write_pos = self._kx_trailing_write_index(stmts)
        if write_pos is None:
            return prog
        loop_info = self._kx_repeat_loop_info(stmts, write_pos)
        if loop_info is not None:
            loop_pos, loop = loop_info
            target_stmts = loop.body.stmts
            limit = self._kx_loop_insert_limit(target_stmts)
            insert_pos = self._draw_uniform_insert_pos(
                target_stmts, limit, rng, prefer_late=True)
            want_uniform = self._kx_point_is_uniform(target_stmts, insert_pos)
            parent_stmts = stmts
            parent_stop = loop_pos
            include_rep = True
        else:
            target_stmts = stmts
            insert_pos = self._draw_uniform_insert_pos(stmts, write_pos, rng)
            want_uniform = self._kx_point_is_uniform(stmts, insert_pos)
            parent_stmts = None
            parent_stop = None
            include_rep = False

        # 2) Reconstruct a GenContext whose pool reflects the in-kx_scope integer
        #    vars at the insertion point, with `_fv` already declared.
        kx_is_2d = self._kx_kernel_is_2d(stmts[write_pos])
        ctx = self._create_ctx(
            prog, kernel, rng, insert_pos, want_uniform, kx_is_2d,
            local_stmts=target_stmts,
            parent_stmts=parent_stmts,
            parent_stop=parent_stop,
            include_rep=include_rep)
        # Names the splice may legitimately REFERENCE (host pool vars, params,
        # tid/_fv): a collision on these is a sampled reference, not a redecl.
        pool_names = {v.name for v in ctx.kx_scope.kx_all_variables()}

        # 3) Weave a small budget of NEW observed statements.
        # Ensure the WeaveBlock REGISTRY is populated (decorators run on
        # import).  Cheap + idempotent.
        import generator_ir.gen.weave_handlers as _wh  # noqa: F401

        ctx.kx_open_block()

        self._render_host_coupling(ctx, rng, pool_names)

        # Seed a couple of DECLARED writable scratch vars into the spliced
        # block.  The host pool is reconstructed READ-ONLY (host vars must not
        # be overwritten), so a woven block that needs an assignable target
        # (kx_random_var(for_write) / kx_sample_or_create(for_write)) would
        # otherwise find none and CREATE a fresh `v_*` whose declaration can
        # escape the captured block -> "identifier v_0 is undefined".  Giving
        # blocks a declared, writable, non-reserved scratch var avoids that
        # path entirely.
        for _ in range(2):
            ctx.fresh(T.u32, init=ctx.kx_random_var(T.u32, recency=0.5))

        budget = max(3, min(10, 3 + rng.randint(0, 6)))
        # Weave with splice-unsafe blocks disabled for the whole pass (incl.
        # any nested weave from control-flow blocks).
        with _insert_blocks_disabled():
            self._kx_weave(ctx, budget)
        new_stmts = ctx.kx_close_block()

        if not new_stmts:
            return prog  # nothing generated -> reject (no-op; pipeline retries)

        # 4) Reject the splice if it would REDECLARE any host name.  Counter
        #    offsetting (see _create_ctx) prevents collisions for fresh-name
        #    vars (cv_/v_/wsmem_*/...), but a template-fragment block could
        #    emit HARDCODED names that collide if the same fragment is already
        #    present.  Rather than a kx_scope-blind rename, reject (return the
        #    ORIGINAL program unchanged) so the pipeline retries with a
        #    different RNG -- safe and rare.  A collision on a POOL name is a
        #    legitimate sampled reference, not a redeclaration.
        host_decls = self._kx_declared_names(kernel.body.stmts)
        new_decls = self._kx_declared_names(new_stmts)
        collisions = (new_decls & host_decls) - pool_names
        if collisions:
            return clone_program(program)

        # 4b) Defensive net: reject the splice if it REFERENCES a fresh-pool
        #     var (cv_/v_/ov_<n>) that is neither a host pool name nor
        #     declared within the splice -- a dangling reference (e.g. a block
        #     that registered a var in kx_scope via kx_sample_or_create but whose
        #     decl escaped the captured block) would otherwise be "identifier
        #     undefined".  Restricted to the unambiguous fresh-pool prefixes
        #     so it never trips on builtins / intrinsics / keywords.
        block = Block(list(new_stmts))
        referenced = {v.name for v in collect(block, lambda n: isinstance(n, Var))
                      if _POOL_REF_RE.fullmatch(v.name)}
        for rc in collect(block, lambda n: isinstance(n, RawCode)):
            referenced.update(_POOL_REF_RE.findall(rc.text or ""))
        if referenced - (pool_names | new_decls):
            return clone_program(program)

        # 5) Splice the new statements in at the chosen point.  Sampled
        #    references to HOST variables keep their original names.
        target_stmts[insert_pos:insert_pos] = new_stmts

        # 6) Propagate any INPUT-buffer enlargement the weave requested back
        #    to the host program config.  A woven strided-unroll statement
        #    reads `input[(tid%n) + j*n]` whose reach is strided_unroll_max*n;
        #    without copying the kx_multiplier back, the harness sizes the input
        #    buffer from the ORIGINAL (smaller) kx_multiplier and the spliced
        #    read goes OUT OF BOUNDS.  Take the max so we never shrink an
        #    already-strided host kernel's buffer.
        woven_mult = int(getattr(ctx.config, "input_size_mult", 1) or 1)
        cur_mult = int(getattr(prog.launch, "input_size_mult", 1) or 1)
        if woven_mult > cur_mult:
            prog.launch.input_size_mult = woven_mult
        return prog

    # perturb_opt / perturb_deopt both just insert more observed work.
    def perturb_opt(self, program: Program, rng: random.Random, param=None) -> Program:
        return self.perturb(program, rng, param)

    def perturb_deopt(self, program: Program, rng: random.Random, param=None) -> Program:
        return self.perturb(program, rng, param)

    # ------------------------------------------------------------------
    # Weaving (weave() minus the FP-island / template blocks)
    # ------------------------------------------------------------------

    @staticmethod
    def _kx_weave(ctx, budget: int) -> None:
        """Weave a small budget of observed integer statements.

        Delegates to the standard ``GenContext.weave`` loop; the caller runs
        this inside ``_insert_blocks_disabled()`` so FP islands and fresh
        template fragments do not fire.

        FP islands need an `output_fp` param/write that a splice cannot add
        safely to an existing signature.  Fresh template fragments often
        redeclare hardcoded scratch names already present in the same fused
        loop kx_scope.  The splice still interacts with existing template code
        by reading its live loop-kx_scope variables."""
        ctx.weave(budget)

    # ------------------------------------------------------------------
    # Insertion-point selection
    # ------------------------------------------------------------------

    @staticmethod
    def _kx_trailing_write_index(stmts) -> Optional[int]:
        """Index of the trailing top-level If that holds the output write."""
        for i in range(len(stmts) - 1, -1, -1):
            if isinstance(stmts[i], If):
                return i
        return None

    @staticmethod
    def _kx_stmt_uniform(stmt) -> bool:
        """Recorded uniformity for a top-level statement (defaults to True
        when no emit_ctx was stamped -- a top-level stmt is uniform unless
        the builder marked it otherwise)."""
        ec = getattr(stmt, "emit_ctx", None)
        if ec is None:
            return True
        return bool(getattr(ec, "uniform", True))

    def _kx_point_is_uniform(self, stmts, pos: int) -> bool:
        """A point is uniform iff the statement that will follow the insertion
        (the one currently at *pos*) is recorded uniform.  Inserting before a
        uniform statement keeps us at uniform control-flow depth."""
        if pos < len(stmts):
            return self._kx_stmt_uniform(stmts[pos])
        return True

    @staticmethod
    def _kx_min_insert_pos(stmts) -> int:
        """Lowest legal insertion index: just AFTER the structural prologue
        decls every woven statement references (``tid`` plus the 2D coords
        ``x``/``y``/``idx`` and ``_fv``).  Inserting before them would emit
        code that reads a variable not yet declared in C kx_scope."""
        prologue = {"tid", "idx", "x", "y", "gid", "_fv"}
        last = 0
        for i, s in enumerate(stmts):
            if isinstance(s, Declare) and s.name in prologue:
                last = i + 1
        return last

    def _draw_uniform_insert_pos(self, stmts, write_pos: int,
                                 rng: random.Random,
                                 prefer_late: bool = False) -> int:
        """Pick a block-level insertion index in ``[min, write_pos]`` after
        the structural prologue.  Prefer a recorded-uniform point so
        sync/shared blocks are admissible; fall back to any valid point."""
        lo = min(self._kx_min_insert_pos(stmts), write_pos)
        if prefer_late and write_pos > lo:
            # For loop-body splices, late insertion means more woven/template
            # locals are in kx_scope, so the new code couples with the body.
            lo = max(lo, write_pos // 3)
        candidates = list(range(lo, write_pos + 1))
        if not candidates:
            return write_pos
        uniform_pts = [p for p in candidates if self._kx_point_is_uniform(stmts, p)]
        pool = uniform_pts if uniform_pts else candidates
        return rng.choice(pool)

    @staticmethod
    def _kx_repeat_loop_info(stmts, write_pos: int):
        """Return ``(index, For)`` for the amplified `_rep` loop if present."""
        for i, s in enumerate(stmts[:write_pos]):
            if (isinstance(s, For) and s.var == "_rep"
                    and isinstance(s.body, Block)):
                return i, s
        return None

    @staticmethod
    def _kx_loop_insert_limit(stmts) -> int:
        """Insertion upper bound inside the repeat loop.

        The loop-carried `_lc_*` block closes the anti-hoist cycle near the
        end of the loop.  Insert before it when possible so spliced
        observations feed the same cycle; otherwise use the loop end."""
        for i, s in enumerate(stmts):
            if isinstance(s, Assign):
                tgt = s.target
                if (isinstance(tgt, Var)
                        and str(getattr(tgt, "name", "")).startswith("_lc_")):
                    return i
        return len(stmts)

    # ------------------------------------------------------------------
    # Kernel-shape detection
    # ------------------------------------------------------------------

    @staticmethod
    def _kx_kernel_is_2d(write_if: If) -> bool:
        """The 2D weave write is the XOR form ``output[idx] = output[idx]^_fv``;
        the 1D write is the plain ``output[tid] = _fv``."""
        then = write_if.then
        if not isinstance(then, Block):
            return False
        for t in then.stmts:
            if _is_fv_output_write(t) and isinstance(getattr(t, "value", None), BinOp):
                return True
        return False

    # ------------------------------------------------------------------
    # GenContext reconstruction
    # ------------------------------------------------------------------

    # Function-kx_scope integer scalars that must NOT enter the readable pool:
    # the off-checksum FP accumulator `output_fp_acc` holds reinterpreted
    # float/double bits (XORed in by kx_observe_fp).  It is an `unsigned int` so
    # it would otherwise be a sampleable READ -- but folding its
    # compiler-divergent FP bits into the integer checksum `_fv` breaks the
    # old == new invariant.  (It is already in GenContext.RESERVED_NAMES for
    # WRITES; this keeps it out of READS too.)
    _POOL_EXCLUDE = frozenset({"output_fp_acc"})
    _COUPLING_EXCLUDE = frozenset({
        "tid", "gid", "n", "output", "input", "_fv", "_rep", "_ci",
        "x", "y", "idx", "output_fp_acc",
    })
    _COUPLING_PREFIXES = (
        "cv_", "v_", "ov_", "_lc_", "lane_", "warp_", "warp_val_",
        "sdata_", "bc_smem_", "wc_smem_", "sh_", "rp_", "cv_buf_", "var_",
    )

    def _render_host_coupling(self, ctx, rng: random.Random, host_pool_names) -> None:
        """Force each accepted splice to read the existing loop/value pool."""
        uint_t = T.u32

        def kx_is_scalar_int(v: Var) -> bool:
            vt = unqualified(v.ty)
            return (not isinstance(vt, (TPointer, TArray))
                    and isinstance(vt, _LOOSE_INT_TYPES))

        candidates = [
            v for v in ctx.kx_scope.kx_all_variables()
            if v.name in host_pool_names
            and v.name not in self._COUPLING_EXCLUDE
            and kx_is_scalar_int(v)
        ]
        if not candidates:
            return

        preferred = [
            v for v in candidates
            if v.name.startswith(self._COUPLING_PREFIXES)
        ]
        pool = preferred if preferred else candidates

        def kx_pick():
            weights = [1.0 + i for i in range(len(pool))]
            return rng.choices(pool, weights=weights, k=1)[0]

        def kx_as_uint(v: Var) -> Cast:
            return Cast(uint_t, to=uint_t, arg=Var(v.ty, name=v.name))

        a = kx_pick()
        expr: Expr = kx_as_uint(a)
        if len(pool) > 1 and rng.random() < 0.75:
            b = kx_pick()
            op = rng.choice((
                Arith.BXOR,
                Arith.ADD,
                Arith.BOR,
                Arith.MUL,
            ))
            expr = BinOp(uint_t, op=op, lhs=expr, rhs=kx_as_uint(b))
        bridge = ctx.fresh(uint_t, init=expr)
        ctx.observe(bridge)

    def _gather_int_decls(self, stmts, stop: int) -> List[Var]:
        """Integer scalar declarations in ``stmts[:stop]`` that are in kx_scope
        for an insertion at that same block level."""
        out: List[Var] = []
        seen = set()
        for i, s in enumerate(stmts):
            if i >= stop:
                break
            if isinstance(s, Declare):
                vt = unqualified(s.gtype)
                if isinstance(vt, (TPointer, TArray)):
                    continue
                if not isinstance(vt, _LOOSE_INT_TYPES):
                    continue
                if s.name in seen or s.name in self._POOL_EXCLUDE:
                    continue
                seen.add(s.name)
                out.append(Var(s.gtype, name=s.name))
        return out

    def _gather_inscope_int_vars(
        self,
        kernel: Func,
        insert_pos: int,
        *,
        local_stmts=None,
        parent_stmts=None,
        parent_stop=None,
        include_rep: bool = False,
    ) -> List[Var]:
        """Integer variables visible at the splice point.

        For loop-body insertion this includes top-level declarations before
        the `_rep` loop, the loop counter itself, and declarations earlier in
        the loop body.  For the fallback top-level insertion it covers the
        declarations before the insertion point."""
        if local_stmts is None:
            return self._gather_int_decls(kernel.body.stmts, insert_pos)

        out: List[Var] = []
        seen = set()

        def add(v: Var):
            if v.name in seen:
                return
            seen.add(v.name)
            out.append(v)

        if parent_stmts is not None and parent_stop is not None:
            for v in self._gather_int_decls(parent_stmts, parent_stop):
                add(v)
        if include_rep:
            add(Var(T.i32, name="_rep"))
        for v in self._gather_int_decls(local_stmts, insert_pos):
            add(v)
        return out

    def _create_ctx(self, prog: Program, kernel: Func, rng: random.Random,
                  insert_pos: int, want_uniform: bool, kx_is_2d: bool, *,
                  local_stmts=None, parent_stmts=None, parent_stop=None,
                  include_rep: bool = False):
        from generator_ir.gen.options import GenConfig
        from generator_ir.gen.context import GenContext

        gcfg = prog.launch
        arch = getattr(gcfg, "arch", "sm_75")
        backend = "hip" if str(arch).lower().startswith("gfx") else "cuda"
        cfg = GenConfig(
            seed=rng.randrange(1 << 30),
            complexity=getattr(gcfg, "complexity", "medium")
            if isinstance(getattr(gcfg, "complexity", None), str) else "medium",
            target=backend, isa=str(arch),
        )
        cfg.allow_volatile = getattr(gcfg, "allow_volatile", False)
        # INPUT-buffer sizing for woven strided reads.  A bare affine strided
        # access `input[(tid%n) + j*n]` (j in [0,U), U <= strided_unroll_max)
        # reaches index < strided_unroll_max*n, so the host input buffer must
        # be at least that large or the read is OUT OF BOUNDS.  Pre-size the
        # kx_multiplier to the full strided reach for 1D kernels, never
        # shrinking below what the host buffer was already sized for;
        # perturb() then propagates the final kx_multiplier back to prog.launch
        # so the harness enlarges the input buffer to match.  (A larger,
        # fully-initialized input buffer leaves every in-range read's value
        # unchanged -> observable result preserved.)
        host_mult = int(getattr(gcfg, "input_size_mult", 1) or 1)
        cfg.input_size_mult = host_mult
        if not kx_is_2d and float(getattr(cfg, "p_strided_unroll", 0.0) or 0.0) > 0.0:
            cfg.input_size_mult = max(
                host_mult, int(getattr(cfg, "strided_unroll_max", 32) or 32))
        ctx = GenContext(cfg)

        # Weave mode + integer-exact observe path.  We set these directly
        # (NOT via bootstrap(), which would emit a fresh `_fv` decl) -- `_fv`
        # already exists in the host kernel.
        ctx._weave_mode = True
        cfg.observable_fold = True
        ctx._fv_declared = True
        ctx.kx_is_2d = kx_is_2d
        # Block geometry: a 2D kernel is NOT 1D, so requires_uniform
        # (sync/shared) blocks must NOT fire (they would race on shared
        # memory).
        ctx._block_is_1d = (not kx_is_2d)

        # Recorded-uniform insertion: if the chosen point is non-uniform,
        # mark the context non-uniform so can_execute() rejects the
        # requires_uniform (sync/shared) blocks -- they only fire at uniform
        # points.
        if not want_uniform:
            ctx.if_depth = 1
            ctx.in_nested_control_flow = True

        # Seed the variable pool: kx_declare the kernel params, `tid` (every
        # weave kernel exposes it), and the in-kx_scope integer locals before
        # the point.
        for p in kernel.params:
            ctx.kx_declare(Var(p.gtype, name=p.name))
        # tid / idx / x / y if present as decls are covered below; ensure tid.
        inscope = self._gather_inscope_int_vars(
            kernel, insert_pos,
            local_stmts=local_stmts,
            parent_stmts=parent_stmts,
            parent_stop=parent_stop,
            include_rep=include_rep)
        have = {v.name for v in inscope}
        if "tid" not in have:
            ctx.kx_declare(Var(T.u32, name="tid"), divergent=True)
        for v in inscope:
            ctx.kx_declare(v, divergent=True)
        # Make `_fv` itself samplable for further mixing (it is a live uint).
        ctx.kx_declare(Var(T.u32, name="_fv"))

        # CRITICAL (splice soundness): every RECONSTRUCTED host variable is
        # READ-ONLY for the weave.  We re-declared the kernel's in-kx_scope ints
        # (and params/tid/_fv) only so woven blocks can READ them; a block
        # must never SAMPLE one as a `for_write` result sink.
        # GenContext.RESERVED_NAMES already protects the structural
        # addressing vars (tid/idx/x/y/output_fp_acc) and i_* loop vars, but
        # NOT any other host addressing var.  If the weave overwrote one of
        # those, a later `smem[base*32+..]` / `output[idx]` would index out
        # of bounds -> GPU OOB.  Shadow the class-level reserved set with an
        # INSTANCE frozenset that additionally reserves EVERY host name we
        # re-declared, so the weave can only write to the FRESH vars it
        # declares itself (cv_*/v_*/...).  Both write paths
        # (kx_random_var(for_write) and kx_sample_or_create(for_write)) consult
        # self.RESERVED_NAMES, so this covers them; loose READS are
        # unaffected -- the host pool stays readable.
        host_pool_names = {v.name for v in ctx.kx_scope.kx_all_variables()}
        ctx.RESERVED_NAMES = frozenset(ctx.RESERVED_NAMES | host_pool_names)

        # Prevent name collisions UP FRONT (instead of kx_scope-blind post-hoc
        # renaming, which would also rewrite sampled references to host
        # vars): offset every fresh-name counter past the host kernel's
        # highest existing `<prefix>_<num>` index, so every var the builder
        # declares (cv_/v_/wsmem_*/loop vars/...) gets a name not already
        # used in the host.
        self._kx_offset_name_counters(ctx, kernel)
        return ctx

    @classmethod
    def _kx_declared_names(cls, stmts) -> set:
        """Names DECLARED (not merely referenced) within *stmts*: Declare
        names plus C declarations embedded in RawCode text."""
        block = Block(list(stmts))
        names = {d.name for d in declarations(block)}
        for rc in collect(block, lambda n: isinstance(n, RawCode)):
            names.update(_VERBATIM_DECL_RE.findall(rc.text or ""))
        return names

    @staticmethod
    def _kx_offset_name_counters(ctx, kernel: Func) -> None:
        """Bump ctx._name_counters[prefix] past the highest host index for
        that prefix, for EVERY ``<prefix>_<num>`` name in the host kernel."""
        max_idx = {}
        for name in all_var_names(kernel):
            m = _NAME_RE.match(name)
            if not m:
                continue
            pref, num = m.group(1), int(m.group(2))
            if pref not in max_idx or num > max_idx[pref]:
                max_idx[pref] = num
        for pref, num in max_idx.items():
            cur = ctx._name_counters.get(pref, 0)
            ctx._name_counters[pref] = max(cur, num + 1)
