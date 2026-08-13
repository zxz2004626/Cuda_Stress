"""Weaving-mode kernel assembly (the two-tier codegen path).

Structure:
    params -> bootstrap -> weave(budget) -> [promoted state + _lc block]
    -> for (_rep < R) { woven body; state updates; _fv mix }
    -> guarded single output write

Anti-collapse machinery:
  * a subset of woven ``cv_*`` scalars is promoted to loop-carried state
    (declaration hoisted out of the repeat loop, rewritten to a read-modify-
    write each iteration);
  * a dedicated ``_lc_*`` accumulator block runs every iteration with
    intrinsics / safe division / casts and cycles through ``_fv``;
  * ``_fv`` is mixed with ``_rep`` so the loop cannot be constant-folded.

Safety: promoted candidates used as memory indices or div/mod bounds are
never promoted (a promoted value could break allocation bounds).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from generator_ir.ir.typedecl import (
    GType, T, MemSpace, Qual, TArray, TPointer, TQualified,
    TUInt, TUShort, TULong, TBool, TInt,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp, Logic
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, Cast, SafeBin,
    Subscript, Call,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, Block, If, For,
)
from generator_ir.ir.program import Func, Program, LaunchConfig
from generator_ir.gen.options import GenConfig
from generator_ir.gen.context import GenContext
from generator_ir.gen.assembler import KernelGenerator


# Names forming the kernel prologue (kept outside the repeat loop).
_PROLOGUE_NAMES = frozenset({"_fv", "output_fp_acc", "tid", "x", "y", "idx"})
# Types eligible for loop-state promotion.
_PROMOTABLE_TYPES = (TUInt, TUShort, TULong, TBool)


def _zero_for(t: GType) -> Lit:
    return Lit(t, value=False if isinstance(t, TBool) else 0)


def _state_budget(ctx: GenContext) -> int:
    return {"low": 6, "medium": 12, "high": 20, "stress": 32}.get(
        getattr(ctx.config, "complexity", "medium"), 12)


def _is_promotable(stmt: Stmt) -> bool:
    if not isinstance(stmt, Declare):
        return False
    if not stmt.name.startswith("cv_") or stmt.init is None:
        return False
    if stmt.space is not MemSpace.GENERIC or stmt.quals:
        return False
    vt = stmt.gtype
    if isinstance(vt, (TQualified, TPointer, TArray)):
        return False
    return isinstance(vt, _PROMOTABLE_TYPES)


def _mark_unsafe(obj, candidates: set, out: set, seen: set) -> None:
    """Mark candidates used as subscript indices or SDIV/SMOD bounds."""
    from generator_ir.ir.traversal import traverse, collect
    # traverse() only understands IR nodes; a bare list of statements has no
    # children, so iterate the statements ourselves (previously the whole
    # filter was silently inert and promoted bounds like `n*n` got rewritten
    # into arbitrary loop-carried state -> OOB reads).
    items = obj if isinstance(obj, (list, tuple)) else [obj]
    for item in items:
        for node in traverse(item):
            if isinstance(node, Subscript):
                for v in collect(node.index, lambda n: isinstance(n, Var)):
                    if v.name in candidates:
                        out.add(v.name)
            elif isinstance(node, SafeBin) and node.op in (SafeOp.SDIV, SafeOp.SMOD):
                for v in collect(node.rhs, lambda n: isinstance(n, Var)):
                    if v.name in candidates:
                        out.add(v.name)


def _promote_loop_state(loop_body: List[Stmt], ctx: GenContext):
    """Hoist a bounded subset of cv_* decls out of the loop; returns
    (hoisted_decls, rewritten_body, promoted_vars)."""
    cand_idx = [i for i, s in enumerate(loop_body) if _is_promotable(s)]
    cand_names = {loop_body[i].name for i in cand_idx}
    unsafe: set = set()
    _mark_unsafe(loop_body, cand_names, unsafe, set())
    cand_idx = [i for i in cand_idx if loop_body[i].name not in unsafe]
    if not cand_idx:
        return [], loop_body, []

    budget = min(len(cand_idx), _state_budget(ctx))
    chosen = (set(ctx.rng.sample(cand_idx, budget))
              if len(cand_idx) > budget else set(cand_idx))

    hoisted: List[Stmt] = []
    promoted: List[Var] = []
    rewritten: List[Stmt] = []
    slot = 0
    for i, stmt in enumerate(loop_body):
        if i not in chosen:
            rewritten.append(stmt)
            continue
        assert isinstance(stmt, Declare)
        predecl = Declare(name=stmt.name, gtype=stmt.gtype,
                          space=stmt.space, init=_zero_for(stmt.gtype))
        predecl.emit_ctx = getattr(stmt, "emit_ctx", None)
        hoisted.append(predecl)
        promoted.append(Var(stmt.gtype, name=stmt.name))
        rewritten.append(_state_reinit(stmt, slot))
        slot += 1
    return hoisted, rewritten, promoted


def _state_reinit(decl: Declare, slot: int) -> Stmt:
    """Read-modify-write for a promoted variable (mixes old state, _rep, _fv)."""
    def U(e: Expr) -> Cast:
        return Cast(T.u32, to=T.u32, arg=e)
    var = Var(decl.gtype, name=decl.name)
    old_mix = BinOp(T.u32, op=Arith.MUL, lhs=U(var),
                    rhs=Lit(T.u32, value=(0x9E3779B1 * (slot + 1)) & 0xFFFFFFFF))
    rep_mix = BinOp(T.u32, op=Arith.MUL, lhs=U(Var(T.i32, name="_rep")),
                    rhs=Lit(T.u32, value=(0x85EBCA6B * (slot + 3)) & 0xFFFFFFFF))
    feedback = BinOp(T.u32, op=Arith.BXOR,
                     lhs=BinOp(T.u32, op=Arith.ADD, lhs=old_mix, rhs=rep_mix),
                     rhs=Var(T.u32, name="_fv"))
    mixed = BinOp(T.u32, op=Arith.BXOR, lhs=U(decl.init), rhs=feedback)
    rhs = mixed if isinstance(decl.gtype, TUInt) else Cast(decl.gtype, to=decl.gtype, arg=mixed)
    stmt = Assign(var, rhs)
    stmt.emit_ctx = getattr(decl, "emit_ctx", None)
    return stmt


def _state_tail(promoted: List[Var]) -> List[Stmt]:
    """End-of-iteration update + _fv fold for promoted state."""
    if not promoted:
        return []
    def U(e: Expr) -> Cast:
        return Cast(T.u32, to=T.u32, arg=e)
    tail: List[Stmt] = []
    fv = Var(T.u32, name="_fv")
    rep_u = U(Var(T.i32, name="_rep"))
    fold = BinOp(T.u32, op=Arith.MUL, lhs=fv, rhs=Lit(T.u32, value=0x9E3779B1))
    for i, v in enumerate(promoted):
        state_u = U(Var(v.ty, name=v.name))
        bump = BinOp(T.u32, op=Arith.BXOR,
                     lhs=state_u,
                     rhs=BinOp(T.u32, op=Arith.ADD,
                               lhs=Var(T.u32, name="_fv"),
                               rhs=BinOp(T.u32, op=Arith.MUL, lhs=rep_u,
                                         rhs=Lit(T.u32, value=(0xC2B2AE35 * (i + 1)) & 0xFFFFFFFF))))
        rhs = bump if isinstance(v.ty, TUInt) else Cast(v.ty, to=v.ty, arg=bump)
        tail.append(Assign(Var(v.ty, name=v.name), rhs))
        fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=state_u)
    tail.append(Assign(fv, fold))
    return tail


def _loop_carried_block(ctx: GenContext, coupled: Optional[List[Var]] = None):
    """The _lc_* accumulator block: K persistent accumulators updated each
    iteration through intrinsics / safe division / casts / helper calls."""
    cx = getattr(ctx.config, "complexity", "medium")
    base_k = {"low": 8, "medium": 14, "high": 22, "stress": 40}.get(cx, 14)
    k = max(4, base_k + ctx.rnd_int(-2, 4))
    palette = [T.u32, T.u32, T.u32, T.u64, T.ushort, T.uchar]
    allow_vol = getattr(ctx.config, "allow_volatile", False)

    def U(v: Expr) -> Cast:
        return Cast(T.u32, to=T.u32, arg=v)

    def OR1(e: Expr) -> Expr:
        return BinOp(T.u32, op=Arith.BOR, lhs=e, rhs=Lit(T.u32, value=1))

    def IC(name: str, arg: Expr) -> Call:
        return Call(T.u32, fn=name, args=[arg], result_ty=T.u32)

    helpers = [h for h in (getattr(ctx, "_device_helpers", None) or [])
               if getattr(h, "params", None)]
    coupled = list(coupled or [])

    accs: List[Tuple[Var, GType]] = []
    pre: List[Stmt] = []
    for i in range(k):
        t = palette[i % len(palette)]
        name = ctx.kx_fresh_name("_lc")
        seed = BinOp(T.u32, op=Arith.BXOR,
                     lhs=BinOp(T.u32, op=Arith.MUL,
                               lhs=Var(T.u32, name="tid"),
                               rhs=Lit(T.u32, value=(0x9E3779B1 * (i + 1)) & 0xFFFFFFFF)),
                     rhs=Lit(T.u32, value=ctx.rnd_int(1, 0xFFFFFFFF)))
        init = seed if isinstance(t, TUInt) else Cast(t, to=t, arg=seed)
        quals = [Qual.VOLATILE] if (allow_vol and ctx.rnd_bool(0.4)) else []
        pre.append(Declare(name=name, gtype=t, init=init, quals=quals))
        accs.append((Var(t, name=name), t))

    loop: List[Stmt] = []
    for i in range(k):
        a, t = accs[i]
        b = accs[(i + 1) % k][0]
        c = accs[(i + 2) % k][0]
        d = accs[(i + 3) % k][0]
        op = ctx.rnd_choice([0, 1, 2, 3, 4] + ([5] if helpers else []))
        if op == 0:      # IMAD + rotate
            s = ctx.rnd_int(1, 31)
            rot = BinOp(T.u32, op=Arith.BOR,
                        lhs=SafeBin(T.u32, op=SafeOp.SSHL, lhs=U(b), rhs=Lit(T.u32, value=s)),
                        rhs=SafeBin(T.u32, op=SafeOp.SSHR, lhs=U(b), rhs=Lit(T.u32, value=32 - s)))
            mixed = BinOp(T.u32, op=Arith.BXOR,
                          lhs=BinOp(T.u32, op=Arith.MUL, lhs=U(a), rhs=Lit(T.u32, value=2654435761)),
                          rhs=rot)
        elif op == 1:    # popc/clz
            mixed = BinOp(T.u32, op=Arith.ADD, lhs=U(a),
                          rhs=BinOp(T.u32, op=Arith.BXOR,
                                    lhs=IC("__popc", U(b)),
                                    rhs=IC("__clz", OR1(U(c)))))
        elif op == 2:    # software division
            mixed = BinOp(T.u32, op=Arith.BXOR,
                          lhs=SafeBin(T.u32, op=SafeOp.SDIV, lhs=U(a), rhs=OR1(U(d))),
                          rhs=SafeBin(T.u32, op=SafeOp.SMOD, lhs=U(b), rhs=OR1(U(c))))
        elif op == 3:    # bit reverse + shift
            mixed = BinOp(T.u32, op=Arith.BXOR,
                          lhs=IC("__brev", U(a)),
                          rhs=SafeBin(T.u32, op=SafeOp.SSHR, lhs=U(b),
                                      rhs=Lit(T.u32, value=ctx.rnd_int(1, 16))))
        elif op == 5:    # device helper call
            h = ctx.rnd_choice(helpers)
            args = [Cast(p.gtype, to=p.gtype, arg=U(accs[(i + 1 + j) % k][0]))
                    for j, p in enumerate(h.params)]
            mixed = BinOp(T.u32, op=Arith.BXOR, lhs=U(a),
                          rhs=Call(T.u32, fn=h.name, args=args, result_ty=T.u32))
        else:            # ffs + add
            mixed = BinOp(T.u32, op=Arith.ADD, lhs=U(a), rhs=IC("__ffs", OR1(U(c))))
        if coupled and i % 4 == 0:
            mixed = BinOp(T.u32, op=Arith.BXOR, lhs=mixed,
                          rhs=U(coupled[(i // 4) % len(coupled)]))
        rhs = mixed if isinstance(t, TUInt) else Cast(t, to=t, arg=mixed)
        loop.append(Assign(Var(t, name=a.name), rhs))

    # Cycle _fv into the first accumulator; fold all back into _fv.
    a0, t0 = accs[0]
    cyc = BinOp(T.u32, op=Arith.BXOR, lhs=U(a0), rhs=Var(T.u32, name="_fv"))
    loop.append(Assign(Var(t0, name=a0.name),
                       cyc if isinstance(t0, TUInt) else Cast(t0, to=t0, arg=cyc)))
    fv = Var(T.u32, name="_fv")
    fold = BinOp(T.u32, op=Arith.MUL, lhs=fv, rhs=Lit(T.u32, value=2654435761))
    for a, _t in accs:
        fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=U(a))
    loop.append(Assign(fv, fold))
    return pre, loop


def wrap_repeat_loop(kernel: Func, repeat: int, ctx: GenContext) -> None:
    """Wrap the woven body in the _rep repeat loop with anti-collapse state."""
    body = kernel.body
    if body is None or repeat <= 1 or not body.stmts:
        return
    stmts = body.stmts
    out_write = stmts[-1]
    if not isinstance(out_write, If):
        return
    mid = stmts[:-1]
    pre = [s for s in mid
           if isinstance(s, Declare) and s.name in _PROLOGUE_NAMES]
    loop_body = [s for s in mid
                 if not (isinstance(s, Declare) and s.name in _PROLOGUE_NAMES)]
    if not loop_body:
        return

    state_pre, loop_body, promoted = _promote_loop_state(loop_body, ctx)
    pre.extend(state_pre)
    coupled = promoted[:min(4, len(promoted))]
    lc_pre, lc_loop = _loop_carried_block(ctx, coupled=coupled)
    pre.extend(lc_pre)
    loop_body.extend(_state_tail(promoted))
    loop_body.extend(lc_loop)

    # _fv ^= (unsigned)_rep * K  -- pins iterations to the counter.
    rep_mix = Assign(
        Var(T.u32, name="_fv"),
        BinOp(T.u32, op=Arith.BXOR,
              lhs=Var(T.u32, name="_fv"),
              rhs=BinOp(T.u32, op=Arith.MUL,
                        lhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name="_rep")),
                        rhs=Lit(T.u32, value=0x9E3779B1))))
    loop_body.append(rep_mix)
    body.stmts = pre + [For(var="_rep", start=Lit(T.i32, value=0),
                            end=Lit(T.i32, value=int(repeat)),
                            step=Lit(T.i32, value=1),
                            body=Block(loop_body)), out_write]


# ---------------------------------------------------------------------------
# Whole-kernel weaving
# ---------------------------------------------------------------------------

def _kx_setup_params(ctx: GenContext):
    """Declare (output, input, n) + scalar params + tid; return handles."""
    out_param = Declare(name="output", gtype=TPointer(T.u32), quals=[Qual.RESTRICT])
    in_param = Declare(name="input", gtype=TPointer(T.u32),
                       quals=[Qual.RESTRICT, Qual.CONST])
    n_param = Declare(name="n", gtype=T.i32)
    output_var = Var(TPointer(T.u32), name="output")
    input_var = Var(TPointer(T.u32), name="input")
    n_var = Var(T.i32, name="n")
    for v in (output_var, input_var, n_var):
        ctx.kx_declare(v)

    params = [out_param, in_param, n_param]
    for i in range(ctx.config.scalar_param_count):
        ptype = ctx.rnd_choice(ctx.declaration_types())
        pname = f"var_{i}"
        params.append(Declare(name=pname, gtype=ptype))
        ctx.kx_declare(Var(ptype, name=pname))

    tid_expr = BinOp(T.u32, op=Arith.ADD,
                     lhs=ThreadBuiltin(T.u32, axis="x", source="threadIdx"),
                     rhs=BinOp(T.u32, op=Arith.MUL,
                               lhs=ThreadBuiltin(T.u32, axis="x", source="blockIdx"),
                               rhs=ThreadBuiltin(T.u32, axis="x", source="blockDim")))
    tid_decl = Declare(name="tid", gtype=T.u32, init=tid_expr)
    tid_var = Var(T.u32, name="tid")
    ctx.kx_declare(tid_var, divergent=True)
    return params, output_var, input_var, n_var, tid_var, tid_decl


def _weave_uses_fragile_256_shape(ctx: GenContext, chosen: str) -> bool:
    """True when block-size variation should stay disabled for correctness."""
    if chosen in {"reduction_block", "scan_inclusive"}:
        return True
    # pure_composable forces one classic template fragment per kernel.  Some of
    # those fragments still carry blockDim.x==256 assumptions, so only vary grid
    # size on this path until each fragment is independently audited.
    fired = getattr(ctx, "fired_blocks", {})
    return any(name.startswith("tmpl_") for name in fired)


def _pick_launch(ctx: GenContext, chosen: str, kx_is_2d: bool):
    vary = bool(getattr(ctx.config, "vary_launch", False))
    if kx_is_2d:
        if vary:
            block = ctx.rnd_choice([(16, 16, 1), (8, 16, 1), (16, 8, 1),
                                     (8, 8, 1), (32, 8, 1)])
            bx, by, _ = block
            grid = ((128 + bx - 1) // bx, (128 + by - 1) // by, 1)
            return block, grid, 128
        return (16, 16, 1), (8, 8, 1), 128
    if not vary:
        return (256, 1, 1), (128, 1, 1), 8192
    if _weave_uses_fragile_256_shape(ctx, chosen):
        bx = 256
        gx = ctx.rnd_choice([64, 96, 128, 192, 256])
    else:
        bx = ctx.rnd_choice([64, 96, 128, 192, 256, 512])
        total = ctx.rnd_choice([4096, 8192, 12288, 16384, 32768])
        gx = max(1, (total + bx - 1) // bx)
    return (bx, 1, 1), (gx, 1, 1), bx * gx


def _choose_weave_template(ctx: GenContext, template_name: Optional[str],
                           registry) -> str:
    """Pick a WeaveTemplate (port of perffuzzer ``_choose_codegen_template``).

    Legacy template names still drive the campaign distribution, but weaving
    now has a controlled chance to use a whole-kernel skeleton when that
    skeleton matches the legacy intent.  Otherwise it stays with
    pure_composable and asks that template to force the legacy fragment in
    the same kx_scope.
    """
    names = set(registry)
    if template_name is not None and template_name in names:
        return template_name

    legacy_forced = {
        "tiled_gemm": "memory_2d",
        "tile_swap": "memory_2d",
        # `prefix_scan` has no safe weave fragment because it owns output
        # indexing.
        "prefix_scan": "scan_inclusive",
    }
    mapped = legacy_forced.get(template_name or "")
    if mapped in names:
        ctx.features_used.add(f"weave_legacy_structural:{template_name}->{mapped}")
        return mapped

    # Probabilistic structural replacements for legacy names that also have
    # same-kx_scope fragments.  Keep pure_composable as the majority path so the
    # stmt/template interaction remains the default campaign shape.
    legacy_prob = {
        "tree_reduce": ("reduction_block", 0.45),
        "reg_spill_probe": ("register_pressure_stress", 0.45),
        "vec_load_probe": ("compute_stress", 0.35),
        "neighbor_sum": ("memory_2d", 0.25),
        "map_transform": ("compute_stress", 0.15),
    }
    if template_name in legacy_prob:
        mapped, prob = legacy_prob[template_name]
        if mapped in names and ctx.rnd_bool(prob):
            ctx.features_used.add(
                f"weave_structural_sample:{template_name}->{mapped}")
            return mapped

    if (getattr(ctx, "use_float_types", False)
            and "fp_stress" in names and ctx.rnd_bool(0.12)):
        ctx.features_used.add("weave_structural_sample:fp_stress")
        return "fp_stress"

    if template_name is None:
        candidates = []
        weights = []
        for name, weight in (
            ("pure_composable", 3.0),
            ("compute_stress", 1.5),
            ("register_pressure_stress", 1.4),
            ("reduction_block", 1.2),
            ("scan_inclusive", 0.9),
            ("fp_stress", 1.0 if getattr(ctx, "use_float_types", False) else 0.0),
            ("memory_2d", 1.0 if getattr(ctx.config, "vary_launch", False) else 0.5),
        ):
            if name in names:
                candidates.append(name)
                weights.append(weight)
        chosen = ctx.rng.choices(candidates, weights=weights, k=1)[0]
        if chosen != "pure_composable":
            ctx.features_used.add(f"weave_structural_sample:{chosen}")
        return chosen

    ctx._preferred_template_fragment = template_name
    return "pure_composable"


_WEAVE_FLAT_STMT_KINDS = frozenset({
    "assign", "vardecl", "compound", "incr_decr", "expr_stmt", "intrinsic",
})


def _weave_phase_budget(ctx: GenContext, phase: str) -> int:
    """Budget (stmt count) for a weave generation phase, scaled by complexity."""
    cx = getattr(ctx.config, "complexity", "medium")
    if phase == "pre":
        return {"low": 6, "medium": 14, "high": 22, "stress": 40}.get(cx, 14)
    if phase == "post":
        return {"low": 4, "medium": 10, "high": 16, "stress": 30}.get(cx, 10)
    return 8


def _generate_weave_phase_stmts(ctx: GenContext, count: int) -> List[Stmt]:
    """Generate a flat (no control flow) mix of diverse stmt types for weaving.

    Randomly draws from vardecl / assign / compound-assign / incr-decr /
    expr-stmt / intrinsic for each slot, giving structural diversity without
    the deep nested loops that would explode runtime inside the _rep repeat
    loop (outer iterations x inner-loop trip-count = hang).
    Port of perffuzzer ``_generate_codegen_phase_stmts``.
    """
    from generator_ir.gen.stmt_builder import StmtGen
    stmts: List[Stmt] = []
    stmt_gen = StmtGen(ctx)
    for _ in range(count):
        ctx._visible_names = ctx.kx_scope.declared_names()
        kind = ctx.rnd_choice(sorted(_WEAVE_FLAT_STMT_KINDS))
        try:
            if kind == "assign":
                s = stmt_gen._assign()
            elif kind == "vardecl":
                s = stmt_gen._vardecl()
            elif kind == "compound":
                s = stmt_gen._compound_assign()
            elif kind == "incr_decr":
                s = stmt_gen._incr_decr()
            elif kind == "expr_stmt":
                s = stmt_gen._expr_stmt()
            elif kind == "intrinsic":
                s = stmt_gen._intrinsic_stmt()
            else:
                s = stmt_gen._vardecl()
        except Exception:
            s = stmt_gen._vardecl()
        ctx._visible_names = None
        if s is not None:
            if isinstance(s, list):
                stmts.extend(s)
            else:
                stmts.append(s)
    return stmts


def generate_weave_kernel(config: GenConfig,
                          template_name: Optional[str] = None) -> Program:
    """Build a kernel through the weaving path (3-phase codegen architecture).

    Phase 1 (pre-template): diverse flat StmtGen statements (vardecl, assign,
    compound, intrinsic, incr/decr) populate the context with richly-typed,
    inter-dependent variables.

    Phase 2 (template): the chosen WeaveTemplate emits its structural compute
    body (reduction tree, compute stress, FP islands, ...) using the Phase 1
    variables already in kx_scope.

    Phase 3 (post-template): more flat statements that USE the template's
    output variables, coupling the template results to the surrounding code.
    """
    # Side-effect registration of every weave block + template registry.
    from generator_ir.gen import weave_blocks as _wb  # noqa: F401
    from generator_ir.gen.weave_templates import TEMPLATE_REGISTRY

    ctx = GenContext(config)
    config.observable_fold = True
    ctx._weave_mode = True

    kg = KernelGenerator(config)
    try:
        arr_decls, arr_info = kg._kx_build_device_arrays(ctx)
        ctx._device_arrays = arr_info
    except Exception:
        arr_decls = []
    try:
        helpers = kg._build_helpers(ctx, weave=True)
        ctx._device_helpers = helpers
    except Exception:
        helpers = []

    if not TEMPLATE_REGISTRY:
        raise RuntimeError("no weave ProgramTemplates registered")
    chosen = _choose_weave_template(ctx, template_name, TEMPLATE_REGISTRY)
    if chosen in {"fp_island", "fp_stress"}:
        ctx.use_float_types = True
    template = TEMPLATE_REGISTRY[chosen]

    params, output_var, input_var, n_var, tid_var, tid_decl = _kx_setup_params(ctx)

    ctx.kx_open_block()
    ctx.emit(tid_decl)
    ctx.bootstrap()

    # --- Phase 1: pre-template diverse flat stmt generation ---
    for s in _generate_weave_phase_stmts(ctx, _weave_phase_budget(ctx, "pre")):
        ctx.emit(s)

    # --- Phase 2: template body ---
    try:
        template.render_body(ctx)
    except NotImplementedError:
        ctx.weave(ctx.kx_budget_for_complexity())

    # --- Phase 3: post-template flat stmt generation ---
    for s in _generate_weave_phase_stmts(ctx, _weave_phase_budget(ctx, "post")):
        ctx.emit(s)

    # Epilogue: guarded observe-write.
    if ctx.used_fp():
        fp_param = Declare(name="output_fp", gtype=TPointer(T.f32),
                           quals=[Qual.RESTRICT])
        params = params + [fp_param]
        output_fp_var: Optional[Var] = Var(TPointer(T.f32), name="output_fp")
    else:
        output_fp_var = None

    n_u = Cast(T.u32, to=T.u32, arg=n_var)
    kx_is_2d = bool(getattr(ctx, "kx_is_2d", False))
    if kx_is_2d:
        x_var = Var(T.u32, name="x")
        y_var = Var(T.u32, name="y")
        write_index = Var(T.u32, name="idx")
        guard = LogicOp(T.boolean, op=Logic.AND,
                        lhs=RelOp(T.boolean, op=Cmp.LT, lhs=x_var, rhs=n_u),
                        rhs=RelOp(T.boolean, op=Cmp.LT, lhs=y_var, rhs=n_u))
    else:
        write_index = tid_var
        guard = RelOp(T.boolean, op=Cmp.LT, lhs=tid_var, rhs=n_u)
    ctx.kx_open_block()
    ctx.render_observe_write(output_var, index_expr=write_index, xor=kx_is_2d)
    if output_fp_var is not None:
        fp_bits = Var(T.u32, name="output_fp_acc")
        fp_val = Call(T.f32, fn="__uint_as_float", args=[fp_bits], result_ty=T.f32)
        ctx.emit(Assign(Subscript(T.f32, array=output_fp_var, index=write_index), fp_val))
    write_stmts = ctx.kx_close_block()
    ctx.emit(If(cond=guard, then=Block(write_stmts)))
    body_stmts = ctx.kx_close_block()

    kernel = Func(name="kernel_main", params=params, ret=T.void,
                  body=Block(body_stmts), quals=[Qual.KERNEL])
    wrap_repeat_loop(kernel, int(getattr(config, "codegen_repeat", 1024)), ctx)

    block, grid, n_elements = _pick_launch(ctx, chosen, kx_is_2d)
    if not getattr(kernel, "launch_bounds", None):
        actual = block[0] * block[1] * block[2]
        choices = [t for t in (256, 384, 512, 768, 1024) if t >= actual] or [1024]
        kernel.launch_bounds = (ctx.rnd_choice(choices), ctx.rnd_choice([1, 2]))

    shared_bytes = 0
    if getattr(ctx, "_needs_dynamic_shared", False):
        actual = block[0] * block[1] * block[2]
        shared_bytes = max(shared_bytes, actual * 4)
    extra_headers = set()
    if getattr(ctx, "_needs_coop_groups", False):
        extra_headers.add("<cooperative_groups.h>")
    if getattr(ctx, "_needs_coop_groups_memcpy_async", False):
        extra_headers.add("<cooperative_groups/memcpy_async.h>")
    if getattr(ctx, "_needs_wmma", False):
        extra_headers.add("<mma.h>")
    if getattr(ctx, "_needs_coop_groups_reduce", False):
        extra_headers.add("<cooperative_groups.h>")
        extra_headers.add("<cooperative_groups/reduce.h>")
        extra_headers.add("<cooperative_groups/scan.h>")

    # INPUT-buffer sizing for woven strided reads.  A bare affine strided
    # access ``input[(tid%n) + j*n]`` (j in [0,U), U <= strided_unroll_depth)
    # reaches index < strided_unroll_depth*n, so the host input buffer must be
    # at least that large or the read is OUT OF BOUNDS.  The pattern can be
    # woven via TWO paths: yield_strided_unroll (which bumps
    # config.input_size_mult itself) AND StmtGen's "strided_unroll" kind
    # selected from a woven tmpl_* fragment / control-flow block (which does
    # NOT bump it).  Cover BOTH by pre-sizing the kx_multiplier to the full
    # strided reach for 1D kernels (mirrors the legacy assembler path), never
    # shrinking below what a block already requested.
    in_mult = max(1, int(getattr(config, "input_size_mult", 1) or 1))
    if not kx_is_2d and float(getattr(config, "prob_strided_unroll", 0.0) or 0.0) > 0.0:
        in_mult = max(in_mult,
                      int(getattr(config, "strided_unroll_depth", 32) or 32))

    launch = LaunchConfig(
        grid=grid, block=block,
        shared_bytes=shared_bytes,
        body_repeat=1,
        n_elements=n_elements,
        kx_is_2d=kx_is_2d,
        observable_fold=True,
        complexity=config.complexity,
        input_size_mult=in_mult,
        arch=config.isa,
        allow_volatile=config.allow_volatile,
        input_dist=getattr(config, "input_dist", "hash"),
        codegen=True,
        extra_headers=extra_headers,
        needs_constant_mem=bool(getattr(ctx, "_needs_constant_mem", False)),
        checksum_only=config.checksum_only,
    )
    prog = Program(launch)
    for d in arr_decls:
        prog.add(d)
    for h in helpers:
        prog.add(h)
    prog.add(kernel)
    return prog
