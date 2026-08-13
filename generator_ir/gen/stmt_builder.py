"""Random statement generator.

Produces declaration/assignment/control-flow/synchronization statements under
the safety policy:

  * barriers only in uniform control flow (never in a helper body);
  * loop trip counts bounded by the ctx.loop_trip_product budget;
  * assignments never target reserved names or loop induction vars;
  * shared-memory and atomic patterns keep bounded, deterministic addresses.
"""

from __future__ import annotations

from typing import List, Optional

from generator_ir.ir.typedecl import (
    GType, T, TBool, TUInt, TInt, TArray, TPointer, MemSpace, Qual,
    Uniformity,
)
from generator_ir.ir.operators import Arith, Cmp, Logic, UnaOp, Compound, IncDec, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, SafeBin, Subscript, Call, Atomic, AddrOf,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, IncDecStmt, ExprStmt, Declare, Block,
    If, For, While, DoWhile, Switch, Case, Return, Break, Continue,
    Barrier, Fence, UnrollHint, RawCode,
)
from generator_ir.ir.arch import Arch
from generator_ir.gen.context import GenContext
from generator_ir.gen.expr_builder import ExprGen, _INT_INTRINSICS


class StmtGen:
    """Weighted random statement builder."""

    def __init__(self, ctx: GenContext, expr_gen: Optional[ExprGen] = None) -> None:
        self.ctx = ctx
        self.expr_gen = expr_gen or ExprGen(ctx)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sync_legal(self) -> bool:
        c = self.ctx
        # A barrier is legal when control flow is uniform: not inside any
        # if/while/for with non-uniform trip, and never inside device helpers.
        return (not c.in_device_helper
                and c.if_depth == 0
                and not c.in_nested_control_flow
                and not c.inside_breakable_loop
                and c.config.enable_barriers)

    def _kx_in_safe_context(self) -> bool:
        """Top-level, converged control flow (perffuzzer _kx_in_safe_context)."""
        c = self.ctx
        return (c.if_depth == 0 and c.loop_depth == 0
                and not c.in_device_helper
                and not c.in_nested_control_flow)

    # -- backend / arch helpers -----------------------------------------

    def _arch(self) -> Arch:
        return Arch.parse(getattr(self.ctx.config, "isa", "sm_75"))

    def _kx_is_cuda(self) -> bool:
        return getattr(self.ctx.config, "target", "hip") == "cuda"

    def _kx_cuda_supported(self, min_sm: int) -> bool:
        a = self._arch()
        return self._kx_is_cuda() and a.kx_is_nvidia and a.version >= min_sm

    def _is_gfx1200(self) -> bool:
        isa = getattr(self.ctx.config, "isa", "")
        return isa.startswith("gfx") and self.ctx.kx_arch_number() == 1200

    def _kx_risky_patterns_enabled(self) -> bool:
        return not self._is_gfx1200()

    # -- verbatim-pattern text helpers ------------------------------------

    def _fetch_existing_uint_name(self) -> str:
        """An existing uint variable name, or threadIdx.x as a fallback."""
        existing = self._pick_target(T.u32)
        return existing.name if existing is not None else "threadIdx.x"

    def _kx_keep_alive(self, sink_name: str, value_code: str) -> str:
        """A C line that forces the compiler to evaluate *value_code*.

        Default: XOR-fold into the observable ``_fv`` accumulator (DCE-proof
        because ``_fv`` reaches the output write).  With allow_volatile (or
        in a device helper where ``_fv`` is not in kx_scope) use a volatile
        local sink instead.
        """
        if (getattr(self.ctx.config, "allow_volatile", False)
                or getattr(self.ctx, "in_device_helper", False)):
            return f"    volatile unsigned int {sink_name} = {value_code};\n"
        return f"    _fv ^= (unsigned int)({value_code});\n"

    def _kx_opaque_true(self, gate_name: str) -> str:
        """A uniform guard flag that is 1 at runtime but opaque to the compiler."""
        if (getattr(self.ctx.config, "allow_volatile", False)
                or getattr(self.ctx, "in_device_helper", False)):
            return f"    volatile unsigned int {gate_name} = 1u;\n"
        return (f"    unsigned int {gate_name} = "
                f"(((unsigned int)n >> 31) ^ 1u);\n")

    def _pick_target(self, gtype: Optional[GType] = None) -> Optional[Var]:
        """An assignable variable (not reserved, not a loop induction var)."""
        cands: List[Var] = []
        for v in self.ctx.kx_scope.kx_all_variables():
            if self.ctx._visible_names is not None and v.name not in self.ctx._visible_names:
                continue
            if v.name in GenContext.RESERVED_NAMES or v.name.startswith("i_"):
                continue
            vt = v.ty
            if isinstance(vt, (TPointer, TArray)):
                continue
            if gtype is not None and type(vt) is not type(gtype):
                continue
            cands.append(v)
        if not cands:
            return None
        return self.ctx.rnd_choice(cands)

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------

    def build_stmts(self, count: int, depth: int = 0) -> List[Stmt]:
        out: List[Stmt] = []
        for _ in range(count):
            if self.ctx.stmt_count >= self.ctx.config.stmt_budget:
                break
            # use-before-def prevention: expressions may only see names
            # declared before this point.
            self.ctx._visible_names = self.ctx.kx_scope.declared_names()
            s = self._draw_stmt(depth)
            self.ctx._visible_names = None
            if s is None:
                continue
            if isinstance(s, list):
                out.extend(s)
                self.ctx.stmt_count += len(s)
            else:
                out.append(s)
                self.ctx.stmt_count += 1
        return out

    def _kind_table(self, depth: int):
        cfg = self.ctx.config
        # Two weight tiers (perffuzzer): stress budgets bias toward variable
        # declarations to reach 200+ live variables.
        if cfg.stmt_budget >= 300:
            table = [
                ("assign", 18.0), ("vardecl", 26.0), ("if", 10.0),
                ("for", 10.0), ("while", 5.0), ("do_while", 4.0),
                ("switch", 4.0), ("compound", 8.0), ("incr_decr", 4.0),
                ("expr_stmt", 4.0), ("atomic", 4.0), ("sync", 4.0),
                ("fence", 3.0), ("pragma", 3.0), ("emi_guard", 4.0),
                ("intrinsic", 3.0),
            ]
        else:
            table = [
                ("assign", 26.0), ("vardecl", 15.0), ("if", 12.0),
                ("for", 12.0), ("while", 4.0), ("do_while", 4.0),
                ("switch", 3.0), ("compound", 9.0), ("incr_decr", 4.0),
                ("expr_stmt", 3.0), ("atomic", 5.0), ("sync", 7.0),
                ("fence", 3.0), ("pragma", 4.0), ("emi_guard", 4.0),
                ("intrinsic", 3.0),
            ]
        # Context gating (depth ceilings, barrier legality, feature toggles).
        if depth >= cfg.nesting_branch_cap:
            table = [(k, w) for k, w in table if k not in ("if", "switch")]
        if depth >= cfg.nesting_loop_cap or not self._loop_budget_ok():
            table = [(k, w) for k, w in table
                     if k not in ("for", "while", "do_while", "pragma")]
        if not self._sync_legal():
            table = [(k, w) for k, w in table if k != "sync"]
        if not (cfg.enable_atomics and not self.ctx.kx_is_2d
                and not self.ctx.in_device_helper
                and {"tid", "output", "n"} <= self.ctx.kx_scope.declared_names()):
            table = [(k, w) for k, w in table if k != "atomic"]
        # observable_fold divergence patterns (1D, kernel kx_scope only).
        if (cfg.observable_fold and not self.ctx.kx_is_2d
                and not self.ctx.in_device_helper
                and {"tid", "input", "n"} <= self.ctx.kx_scope.declared_names()):
            p_dsw = max(0.0, float(getattr(cfg, "prob_divergent_switch", 0.0)))
            p_sun = max(0.0, float(getattr(cfg, "prob_strided_unroll", 0.0)))
            if p_dsw > 0.0:
                table.append(("divergent_switch", 12.0 * p_dsw))
            if p_sun > 0.0:
                table.append(("strided_unroll", 12.0 * p_sun))
        # DISABLED (mirrors the original's decision): device-array writes use
        # per-thread random indices -> overlapping writes race, making the
        # array contents (and any checksum reading them) non-deterministic.
        # if (self._device_arrays_present()
        #         and not getattr(self.ctx.config, "observable_fold", False)
        #         and not getattr(self.ctx, "_weave_mode", False)):
        #     table.append(("device_array_write", 4.0))
        # Risky cross-thread patterns: verbatim cooperative-groups / warp /
        # pipeline fragments.  Excluded under observable_fold (their folded
        # values can depend on the dynamic active set -> FALSE_MISMATCH) and
        # inside device helpers (kernel-kx_scope names are required).
        if (self._kx_risky_patterns_enabled() and self._kx_in_safe_context()
                and not getattr(cfg, "observable_fold", False)):
            risky = [
                ("sync_predicate", 1.0), ("dynamic_shared", 1.0),
                ("coop_groups", 1.0), ("async_copy", 1.0),
                ("scoped_atomic", 1.0), ("pipeline", 1.0),
                ("warp_aware", 1.0), ("coop_subwarp", 1.0),
                ("coalesced", 1.0), ("gated_barrier", 1.0),
            ]
            if self._kx_is_cuda():
                if self._kx_cuda_supported(70):
                    risky.extend([("cuda_syncwarp", 1.0), ("cuda_match", 1.0),
                                  ("coalesced", 1.0)])  # extra CUDA weight
                if self._kx_cuda_supported(80):
                    risky.extend([("cuda_cache_store", 1.0),
                                  ("cuda_pipeline", 1.0), ("cuda_barrier", 1.0)])
            in_kernel = (self.ctx.kx_scope.lookup("output") is not None
                         and self.ctx.kx_scope.lookup("tid") is not None)
            if not in_kernel:
                drop = {"coop_groups", "async_copy", "scoped_atomic",
                        "pipeline", "warp_aware", "sync_predicate",
                        "dynamic_shared", "coop_subwarp", "coalesced",
                        "gated_barrier", "cuda_barrier"}
                risky = [(k, w) for k, w in risky if k not in drop]
            table.extend(risky)
        # Fuzzilli-style template fusion: splice a 1D template body as a
        # composable fragment at this program point (fold mode only).
        if (getattr(cfg, "observable_fold", False) and not self.ctx.kx_is_2d
                and not getattr(self.ctx, "_in_template_fragment", False)
                and self._kx_in_safe_context()):
            table.append(("template", 4.0))
        return table

    def _draw_stmt(self, depth: int) -> Optional[Stmt]:
        table = self.ctx.weights.apply(
            "stmt", self._kind_table(depth), protected=("assign", "vardecl"))
        names = [k for k, _ in table]
        wts = [w for _, w in table]
        kind = self.ctx.rng.choices(names, weights=wts, k=1)[0]
        fn = {
            "vardecl": self._vardecl,
            "assign": self._assign,
            "compound": self._compound_assign,
            "incr_decr": self._incr_decr,
            "if": lambda: self._if(depth),
            "for": lambda: self._for(depth),
            "while": lambda: self._while(depth),
            "do_while": lambda: self._do_while(depth),
            "switch": lambda: self._switch(depth),
            "sync": self._sync,
            "atomic": self._atomic,
            "fence": self._fence,
            "expr_stmt": self._expr_stmt,
            "pragma": lambda: self._pragma(depth),
            "emi_guard": lambda: self._emi_guard(depth),
            "intrinsic": self._intrinsic_stmt,
            "sync_predicate": self._sync_predicate,
            "dynamic_shared": self._dynamic_shared,
            "coop_groups": self._coop_groups,
            "async_copy": self._async_copy,
            "scoped_atomic": self._scoped_atomic,
            "pipeline": self._pipeline,
            "warp_aware": self._warp_aware,
            "coop_subwarp": self._coop_subwarp,
            "coalesced": self._coalesced_threads,
            "gated_barrier": self._gated_barrier,
            "cuda_syncwarp": self._cuda_syncwarp,
            "cuda_match": self._cuda_match,
            "cuda_cache_store": self._cuda_cache_store,
            "cuda_pipeline": self._cuda_pipeline,
            "cuda_barrier": self._cuda_barrier,
            "template": self._template_fragment,
            "divergent_switch": lambda: self._divergent_switch(depth),
            "strided_unroll": lambda: self._strided_unroll(depth),
            "device_array_write": self._device_array_write,
        }.get(kind, self._assign)
        try:
            return fn()
        except Exception:
            # A failed sub-generation must never break the whole kernel.
            return self._assign()

    def _loop_budget_ok(self) -> bool:
        return self.ctx.loop_trip_product < self.ctx.max_loop_trip_product

    def _device_arrays_present(self) -> bool:
        return bool(getattr(self.ctx, "_device_arrays", None))

    # ------------------------------------------------------------------
    # leaf statements
    # ------------------------------------------------------------------

    def _vardecl(self) -> Stmt:
        t = self.ctx.rnd_choice(self.ctx.declaration_types())
        name = self.ctx.kx_fresh_name("v")
        # Cross-type initializer: init of a different type with an implicit
        # conversion (stresses integer promotion / narrowing passes).
        if (getattr(self.ctx.config, "use_diverse_types", False)
                and self.ctx.rnd_bool(0.25)
                and (t.is_int or t.is_fp)):
            alt = [at for at in self.ctx.kx_computation_types()
                   if type(at) is not type(t)]
            if alt:
                init_t = self.ctx.rnd_choice(alt)
                init = Cast(t, to=t,
                            arg=self.expr_gen.kx_generate(init_t, depth=0),
                            soft=True)
                self.ctx.kx_declare(Var(t, name=name))
                return Declare(name=name, gtype=t, init=init)
        init = self.expr_gen.kx_generate(t, depth=0)
        decl = Declare(name=name, gtype=t, init=init)
        self.ctx.kx_declare(Var(t, name=name))
        return decl

    def _assign(self) -> Stmt:
        # Try every available type for an assignable target (diverse types).
        avail = self.ctx.declaration_types()
        if len(avail) > 2:
            types_to_try = list(avail)
            self.ctx.rng.shuffle(types_to_try)
            for t in types_to_try:
                target = self._pick_target(t)
                if target is None:
                    continue
                # Cross-type RHS: implicit conversion stresses type lowering.
                if (getattr(self.ctx.config, "use_diverse_types", False)
                        and self.ctx.rnd_bool(0.35)
                        and (t.is_int or t.is_fp)):
                    alt = [at for at in self.ctx.kx_computation_types()
                           if type(at) is not type(t)]
                    if alt:
                        rhs_t = self.ctx.rnd_choice(alt)
                        rhs = Cast(t, to=t,
                                   arg=self.expr_gen.kx_generate(rhs_t, depth=0),
                                   soft=True)
                        return Assign(Var(t, name=target.name), rhs)
                return Assign(target, self.expr_gen.kx_generate(t, depth=1))
            return self._vardecl()
        target = self._pick_target()
        if target is None:
            return self._vardecl()
        value = self.expr_gen.kx_generate(target.ty, depth=1)
        return Assign(target, value)

    def _compound_assign(self) -> Stmt:
        avail = self.ctx.declaration_types()
        if len(avail) > 2:
            types_to_try = list(avail)
            self.ctx.rng.shuffle(types_to_try)
            for t in types_to_try:
                if not (t.is_int or t.is_fp):
                    continue
                target = self._pick_target(t)
                if target is not None:
                    return self._compound_on(t, target)
            return self._vardecl()
        target = self._pick_target()
        if target is None:
            return self._vardecl()
        return self._compound_on(target.ty, target)

    def _compound_on(self, t: GType, target: Var) -> Stmt:
        if t.is_int:
            op = self.ctx.rnd_choice([Compound.ADD, Compound.SUB, Compound.MUL,
                                       Compound.BAND, Compound.BOR, Compound.BXOR])
        else:
            op = self.ctx.rnd_choice([Compound.ADD, Compound.SUB, Compound.MUL])
        # Cross-type RHS: implicit conversion (stresses type-lowering passes).
        rhs: Optional[Expr] = None
        if (getattr(self.ctx.config, "use_diverse_types", False)
                and self.ctx.rnd_bool(0.25)
                and (t.is_int or t.is_fp)):
            alt = [at for at in self.ctx.kx_computation_types()
                   if type(at) is not type(t)]
            if alt:
                rhs_t = self.ctx.rnd_choice(alt)
                rhs = Cast(t, to=t,
                           arg=self.expr_gen.kx_generate(rhs_t, depth=0),
                           soft=True)
        # Signed int +=/-=/*= can overflow (UB); rewrite as a wrapped plain
        # assign through the unsigned SafeOp macros so behavior stays defined.
        # FP types must NOT be routed through the integer SafeOp macros
        # (they reinterpret operands as unsigned long long), so gate on int.
        if t.is_int and t.kx_is_signed and op in (Compound.ADD, Compound.SUB, Compound.MUL):
            safe = {Compound.ADD: SafeOp.SADD,
                    Compound.SUB: SafeOp.SSUB,
                    Compound.MUL: SafeOp.SMUL}[op]
            # The cross-type rhs above may be FP (or contain FP libcalls);
            # feeding it into the integer SafeOp macros would corrupt values
            # and put e.g. sin() inside __typeof__ (breaks nvcc 11.5).  The
            # SafeBin branch therefore always uses a fresh same-type rhs.
            # NOTE: `target` must stay a side-effect-free plain Var lvalue --
            # its name appears in both the Assign lhs and the SafeBin lhs.
            rhs = self.expr_gen.kx_generate(t, depth=1)
            return Assign(target, SafeBin(t, op=safe, lhs=Var(t, name=target.name), rhs=rhs))
        if rhs is None:
            rhs = self.expr_gen.kx_generate(t, depth=1)
        return CompoundAssign(target, op, rhs)

    def _incr_decr(self) -> Stmt:
        target = self._pick_target()
        # ++/-- on a signed int overflows at its limit (UB); restrict to
        # unsigned targets where wraparound is defined.
        if (target is None or not target.ty.is_int
                or isinstance(target.ty, TBool) or target.ty.kx_is_signed):
            return self._assign()
        op = self.ctx.rnd_choice(list(IncDec))
        prefix = self.ctx.rnd_bool()
        return IncDecStmt(target, op, prefix)

    def _expr_stmt(self) -> Stmt:
        """A standalone expression statement (dead-code / DCE stress)."""
        types = self.ctx.kx_computation_types()
        if not types:
            return self._vardecl()
        t = self.ctx.rnd_choice(types)
        return ExprStmt(self.expr_gen.kx_generate(t, depth=1))

    def _pragma(self, depth: int) -> List[Stmt]:
        """#pragma unroll [factor] immediately followed by a for loop."""
        factor = self.ctx.rnd_choice([None, 2, 4, 8])
        loop = self._for(depth)
        if isinstance(loop, For):
            return [UnrollHint(factor=factor), loop]
        return [loop]

    def _emi_guard(self, depth: int) -> Stmt:
        """EMI dead-code guard (yarpgen-edit pattern).

        ``if (threadIdx.x >= blockDim.x) { dead code }`` -- always false at
        runtime, but proving it needs range analysis.  Stresses dead code
        elimination and constant propagation.
        """
        cond = RelOp(T.boolean, op=Cmp.GE,
                     lhs=ThreadBuiltin(T.u32, axis="x", source="threadIdx"),
                     rhs=ThreadBuiltin(T.u32, axis="x", source="blockDim"))
        name = self.ctx.kx_fresh_name("_dead")
        body = [
            Declare(name=name, gtype=T.u32, init=Lit(T.u32, value=0xDEADDEAD)),
            Assign(Var(T.u32, name=name),
                   self.expr_gen.kx_generate(T.u32, depth=depth + 1)),
        ]
        return If(cond=cond, then=Block(body))

    def _intrinsic_stmt(self) -> Stmt:
        """An integer intrinsic call statement (yarpgen-edit pattern).

        ``uint var = __popc(expr);`` or ``existing ^= __clz(expr);`` --
        stresses instruction selection for non-standard ALU ops.
        """
        scalar_ints = [t for t in self.ctx.declaration_types()
                       if t.is_int and not isinstance(t, TBool)]
        if not scalar_ints:
            scalar_ints = [T.u32]
        t = self.ctx.rnd_choice(scalar_ints)
        name, aty, rty = self.ctx.rnd_choice(_INT_INTRINSICS)
        arg = Cast(aty, to=aty, arg=self.expr_gen.kx_generate(aty, depth=1),
                   soft=True)
        call = Call(rty, fn=name, args=[arg], result_ty=rty)
        cast = Cast(t, to=t, arg=call, soft=True)
        existing = self._pick_target(t)
        if existing is not None and self.ctx.rnd_bool(0.6):
            mix = BinOp(t, op=Arith.BXOR, lhs=Var(t, name=existing.name),
                        rhs=cast)
            return Assign(Var(t, name=existing.name), mix)
        vname = self.ctx.kx_fresh_name("_intr")
        self.ctx.kx_declare(Var(t, name=vname), divergent=True)
        return Declare(name=vname, gtype=t, init=cast)

    def _fence(self) -> Stmt:
        self.ctx.features_used.add("threadfence")
        if self._is_gfx1200():
            kx_scope = self.ctx.rnd_choice(["device", "block"])
        elif self.ctx.rnd_bool(0.15):
            kx_scope = "system"
        else:
            kx_scope = self.ctx.rnd_choice(["device", "block"])
        return Fence(kx_scope=kx_scope)

    def _sync(self) -> Stmt:
        """A __syncthreads(), possibly flanked by shared-memory ops.

        30% predicate sync (__syncthreads_count/and/or), 10% dynamic
        extern __shared__ pattern, 35% shared write/barrier/read pattern
        (only when risky patterns are enabled); otherwise a bare barrier.
        """
        if not self._sync_legal():
            return self._assign()
        self.ctx.features_used.add("sync")
        if self._kx_risky_patterns_enabled():
            if self.ctx.rnd_bool(0.30):
                return self._sync_predicate()
            if self.ctx.rnd_bool(0.10):
                return self._dynamic_shared()
            if self.ctx.rnd_bool(0.35):
                return self._shared_sync_pattern()
        return Barrier()

    def _shared_sync_pattern(self) -> List[Stmt]:
        """Shared-memory write / barrier / read sequence (bank-conflict and
        register-allocation stress)."""
        stmts: List[Stmt] = []
        name = self.ctx.kx_fresh_name("_shm")
        # Must span the largest block (vary_launch 1D caps at 512) so the
        # pre-barrier write smem[tid % size] is collision-free across warps;
        # an undersized array makes the synced value racy and nondeterministic.
        size = 512
        at = TArray(T.u32, size)
        stmts.append(Declare(name=name, gtype=at, space=MemSpace.SHARED,
                             quals=[Qual.SHARED_MEM]))
        lane = ThreadBuiltin(T.u32, axis="x", source="threadIdx")
        arr = Var(at, name=name)
        w_idx = SafeBin(T.u32, op=SafeOp.SMOD, lhs=lane,
                        rhs=Lit(T.u32, value=size))
        stmts.append(Assign(Subscript(T.u32, array=arr, index=w_idx),
                            self.expr_gen.kx_generate(T.u32, depth=1)))
        stmts.append(Barrier())
        r_idx = SafeBin(T.u32, op=SafeOp.SMOD,
                        lhs=BinOp(T.u32, op=Arith.BXOR, lhs=lane,
                                  rhs=Lit(T.u32, value=1)),
                        rhs=Lit(T.u32, value=size))
        read = Subscript(T.u32, array=arr, index=r_idx)
        target = self._pick_target(T.u32)
        if target is not None:
            stmts.append(Assign(Var(T.u32, name=target.name),
                                BinOp(T.u32, op=Arith.BXOR,
                                      lhs=Var(T.u32, name=target.name),
                                      rhs=read)))
        else:
            vname = self.ctx.kx_fresh_name("_sr")
            self.ctx.kx_declare(Var(T.u32, name=vname), divergent=True)
            stmts.append(Declare(name=vname, gtype=T.u32, init=read))
        return stmts

    def _sync_predicate(self) -> Stmt:
        """A __syncthreads_count/and/or() predicate sync that returns a
        result (warp-level predicate aggregation stress).  Uses simple
        thread-index predicates so the verbatim text is always valid C."""
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("sync_predicate")
        variant = self.ctx.rnd_choice(["count", "and", "or"])
        var_name = self.ctx.kx_fresh_name("_sp")
        predicates = [
            "threadIdx.x < 128",
            "threadIdx.x < 64",
            "threadIdx.x < 32",
            "threadIdx.x % 2 == 0",
            "blockIdx.x > 0",
            "threadIdx.x >= blockDim.x / 2",
        ]
        existing = self._pick_target(T.u32)
        if existing is not None:
            predicates.extend([
                f"{existing.name} > 0u",
                f"{existing.name} != 42u",
                f"({existing.name} & 1u) == 0u",
            ])
        pred = self.ctx.rnd_choice(predicates)
        fn = {"count": "__syncthreads_count",
              "and": "__syncthreads_and",
              "or": "__syncthreads_or"}[variant]
        # Declare in kx_scope so the variable can be referenced later.
        self.ctx.kx_declare(Var(T.i32, name=var_name))
        return RawCode(text=f"int {var_name} = {fn}({pred});")

    def _dynamic_shared(self) -> Stmt:
        """A dynamic extern __shared__ write/sync/read pattern.

        Sets ctx._needs_dynamic_shared so the launch picks up a shared
        memory size argument.  Kept as ONE opaque statement so mutators
        cannot split the extern declaration from its uses.
        """
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("dynamic_shared")
        smem = self.ctx.kx_fresh_name("_dsmem")
        self.ctx._needs_dynamic_shared = True
        write_val = "threadIdx.x"
        existing = self._pick_target(T.u32)
        if existing is not None:
            write_val = existing.name
        target = self._pick_target(T.u32)
        if target is None:
            return self._vardecl()
        return RawCode(text=(
            f"extern __shared__ unsigned int {smem}[];\n"
            f"{smem}[threadIdx.x] = {write_val};\n"
            "__syncthreads();\n"
            f"{target.name} ^= {smem}[threadIdx.x ^ 1];"))

    # ------------------------------------------------------------------
    # control flow
    # ------------------------------------------------------------------

    def _if(self, depth: int) -> Stmt:
        cond = self.expr_gen._comparison(T.boolean, depth + 1)
        # Scale body size with the remaining budget for denser kernels.
        max_body = 8 if self.ctx.config.stmt_budget >= 300 else 4
        self.ctx.if_depth += 1
        self.ctx.in_nested_control_flow = True
        self.ctx.kx_push_scope()
        then_stmts = self.build_stmts(self.ctx.rnd_int(1, max_body), depth + 1)
        self.ctx.kx_pop_scope()
        else_block = None
        if self.ctx.rnd_bool(0.4):
            self.ctx.kx_push_scope()
            else_stmts = self.build_stmts(
                self.ctx.rnd_int(1, max(max_body - 1, 2)), depth + 1)
            self.ctx.kx_pop_scope()
            else_block = Block(else_stmts)
        self.ctx.if_depth -= 1
        self.ctx.in_nested_control_flow = self.ctx.if_depth > 0
        return If(cond=cond, then=Block(then_stmts), otherwise=else_block)

    def _for(self, depth: int) -> Stmt:
        # Bound the trip count by the remaining budget.
        remaining = max(2, self.ctx.max_loop_trip_product
                        // max(1, self.ctx.loop_trip_product))
        if self.ctx.config.stmt_budget >= 300:
            # Stress mode: cap trip counts (many loops with 300+ statements).
            trips = self.ctx.rnd_choice([2, 4, 4, 8, 8])
        else:
            trips = self.ctx.rnd_choice([4, 8, 16, 32, 64])
        trips = max(2, min(trips, remaining))
        # observable_fold: a data-dependent bound blocks full unrolling, so the
        # two compilers' trip-count/unroll heuristics diverge.  Generated BEFORE
        # the induction variable is declared, keeping the bound loop-invariant.
        runtime_end = None
        if (getattr(self.ctx.config, "observable_fold", False) and trips >= 2
                and self.ctx.rnd_bool(getattr(
                    self.ctx.config, "prob_runtime_loop_bound", 0.0))):
            mask = 7 if trips >= 8 else (trips - 1)
            runtime_end = BinOp(T.i32, op=Arith.ADD,
                lhs=BinOp(T.i32, op=Arith.BAND,
                    lhs=Cast(T.i32, to=T.i32,
                             arg=self.expr_gen.kx_generate(T.u32, depth + 1)),
                    rhs=Lit(T.i32, value=mask)),
                rhs=Lit(T.i32, value=1))
        var = self.ctx.kx_fresh_name("i")
        self.ctx.kx_push_scope()
        self.ctx.kx_declare(Var(T.i32, name=var))
        self.ctx.loop_depth += 1
        self.ctx.loop_trip_product *= trips
        prev_breakable = self.ctx.inside_breakable_loop
        prev_nested = self.ctx.in_nested_control_flow
        self.ctx.inside_breakable_loop = True
        self.ctx.in_nested_control_flow = True  # prevent barrier emission inside for
        max_body = 8 if self.ctx.config.stmt_budget >= 300 else 4
        body = self.build_stmts(self.ctx.rnd_int(1, max_body), depth + 1)
        # Conditional break / continue appended at the END of the body so
        # they never reference variables declared later in the same kx_scope.
        if self.ctx.rnd_bool(0.3) and body:
            body.append(If(cond=self.expr_gen._comparison(T.boolean, depth + 1),
                           then=Block([Break()])))
        if self.ctx.rnd_bool(0.15) and body:
            body.append(If(cond=self.expr_gen._comparison(T.boolean, depth + 1),
                           then=Block([Continue()])))
        self.ctx.inside_breakable_loop = prev_breakable
        self.ctx.in_nested_control_flow = prev_nested
        self.ctx.loop_trip_product //= max(1, trips)
        self.ctx.loop_depth -= 1
        self.ctx.kx_pop_scope()
        return For(var=var, start=Lit(T.i32, value=0),
                   end=runtime_end if runtime_end is not None else Lit(T.i32, value=trips),
                   step=Lit(T.i32, value=1),
                   body=Block(body))

    def _bounded_counter_loop(self, depth: int, prefix: str, do_while: bool) -> Stmt:
        """Shared builder for bounded while / do-while counter loops."""
        var = self.ctx.kx_fresh_name(prefix)
        self.ctx.kx_push_scope()
        # The counter is deliberately NOT declared into the variable kx_scope
        # (perffuzzer parity): body statements kx_pick write targets from the
        # kx_scope, so a visible counter could be clobbered by a generated
        # assignment, turning the bounded trip count into a data-dependent,
        # effectively unbounded loop (observed: helper do-while spinning
        # 2^31 iterations -> campaign timeouts).
        decl = Declare(name=var, gtype=T.i32, init=Lit(T.i32, value=0))
        remaining = max(2, self.ctx.max_loop_trip_product
                        // max(1, self.ctx.loop_trip_product))
        if self.ctx.config.stmt_budget >= 300:
            bound = self.ctx.rnd_choice([2, 3, 4, 6, 8])
        else:
            bound = self.ctx.rnd_choice([2, 3, 4])
        bound = max(2, min(bound, remaining))
        self.ctx.loop_trip_product *= bound
        self.ctx.loop_depth += 1
        prev_breakable = self.ctx.inside_breakable_loop
        prev_nested = self.ctx.in_nested_control_flow
        self.ctx.inside_breakable_loop = True
        self.ctx.in_nested_control_flow = True  # prevent barrier emission inside loop
        max_body = 6 if self.ctx.config.stmt_budget >= 300 else 4
        body_stmts = self.build_stmts(self.ctx.rnd_int(1, max_body), depth + 1)
        self.ctx.inside_breakable_loop = prev_breakable
        self.ctx.in_nested_control_flow = prev_nested
        self.ctx.loop_depth -= 1
        self.ctx.loop_trip_product //= max(1, bound)
        body_stmts.append(IncDecStmt(Var(T.i32, name=var), IncDec.INC, False))
        self.ctx.kx_pop_scope()
        cond = RelOp(T.boolean, op=Cmp.LT,
                     lhs=Var(T.i32, name=var), rhs=Lit(T.i32, value=bound))
        if do_while:
            loop: Stmt = DoWhile(body=Block(body_stmts), cond=cond)
        else:
            loop = While(cond=cond, body=Block(body_stmts))
        return Block([decl, loop])

    def _while(self, depth: int) -> Stmt:
        # Emit as a bounded loop over a generated induction variable to
        # keep trip counts finite and deterministic.  Clamped to the
        # remaining trip budget like _for.
        return self._bounded_counter_loop(depth, "w", do_while=False)

    def _do_while(self, depth: int) -> Stmt:
        """A bounded do-while loop with an internal counter."""
        return self._bounded_counter_loop(depth, "dw", do_while=True)

    def _switch(self, depth: int) -> Stmt:
        disc_t = T.u32
        disc = self.expr_gen.kx_generate(disc_t, depth + 1)
        n_cases = self.ctx.rnd_int(2, 4)
        cases: List[Case] = []
        # A switch on a possibly-varying discriminant is divergent control
        # flow: warp votes and barriers are illegal inside the arms.
        self.ctx.if_depth += 1
        self.ctx.in_nested_control_flow = True
        for i in range(n_cases):
            self.ctx.kx_push_scope()
            body = self.build_stmts(self.ctx.rnd_int(1, 2), depth + 1)
            self.ctx.kx_pop_scope()
            cases.append(Case(value=Lit(T.i32, value=i), body=Block(body)))
        self.ctx.kx_push_scope()
        default_body = self.build_stmts(1, depth + 1)
        self.ctx.kx_pop_scope()
        cases.append(Case(value=None, body=Block(default_body)))
        self.ctx.if_depth -= 1
        self.ctx.in_nested_control_flow = self.ctx.if_depth > 0
        # Reduce the discriminant into [0, n_cases) with MODULO, not a bitmask:
        # `disc & (n_cases-1)` only covers every case when n_cases is a power of
        # two, but n_cases may be 3, which would make `disc & 2` skip case 1.
        return Switch(expr=BinOp(T.u32, op=Arith.MOD, lhs=disc,
                                 rhs=Lit(T.u32, value=n_cases)),
                      cases=cases)

    # ------------------------------------------------------------------
    # GPU-flavoured statements
    # ------------------------------------------------------------------

    def _atomic(self) -> Stmt:
        """atomic<op> on the kernel's own output slot (same-thread, safe).

        Guarded by ``if (tid < n)``; the slot index stays SMOD-bounded so
        the address is always in bounds even for degenerate launches.
        """
        self.ctx.features_used.add("atomics")
        tid = Var(T.u32, name="tid")
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        slot = Subscript(T.u32, array=Var(TPointer(T.u32), name="output"),
                         index=SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid, rhs=n_u),
                         locked=True)
        value = self.expr_gen.kx_generate(T.u32, depth=1)
        what = self.ctx.rnd_choice(["Add", "Xor", "Or", "And", "Min", "Max"])
        atomic = ExprStmt(Atomic(T.u32, what=what, addr=slot, operand=value))
        return If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                  then=Block([atomic]))

    def _device_array_write(self) -> Stmt:
        arrays = getattr(self.ctx, "_device_arrays", None)
        if not arrays:
            return self._vardecl()
        info = self.ctx.rnd_choice(arrays)
        node: Expr = Var(info["type"], name=info["name"])
        rem: GType = info["type"]
        for dim in info["dims"]:
            idx = SafeBin(T.u32, op=SafeOp.SMOD,
                          lhs=self.expr_gen.kx_generate(T.u32, depth=2),
                          rhs=Lit(T.u32, value=dim))
            nxt = rem.elem if isinstance(rem, TArray) else T.u32
            node = Subscript(nxt, array=node, index=idx)
            rem = nxt
        val = Cast(rem, to=rem, arg=self.expr_gen.kx_generate(T.u32, depth=1))
        return Assign(node, val)

    # ------------------------------------------------------------------
    # GPU feature patterns (verbatim fragments)
    # ------------------------------------------------------------------

    def _coop_groups(self) -> Stmt:
        """Cooperative groups tiled partition with butterfly reduction."""
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("coop_groups")
        self.ctx._needs_coop_groups = True
        lane = self.ctx.kx_fresh_name("cg_lane")
        val = self.ctx.kx_fresh_name("cg_val")
        blk = self.ctx.kx_fresh_name("cg_block")
        warp = self.ctx.kx_fresh_name("cg_warp")
        ivar = self.ctx.kx_fresh_name("cg_i")
        sink = self.ctx.kx_fresh_name("cg_sink")
        src_val = self._fetch_existing_uint_name()
        code = (
            f"{{ // Cooperative groups tiled partition\n"
            f"    namespace cg = cooperative_groups;\n"
            f"    cg::thread_block {blk} = cg::this_thread_block();\n"
            f"    cg::thread_block_tile<32> {warp} = cg::tiled_partition<32>({blk});\n"
            f"    unsigned int {lane} = {warp}.thread_rank();\n"
            f"    unsigned int {val} = {src_val};\n"
            f"    for (int {ivar} = {warp}.size() / 2; {ivar} > 0; {ivar} /= 2) {{\n"
            f"        {val} += {warp}.shfl_xor({val}, {ivar});\n"
            f"    }}\n"
            f"{self._kx_keep_alive(sink, val)}"
            f"}}"
        )
        return RawCode(text=code)

    def _async_copy(self) -> Stmt:
        """Cooperative-groups-based shared memory fill with a block sync."""
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("async_copy")
        self.ctx._needs_coop_groups = True
        buf = self.ctx.kx_fresh_name("async_buf")
        blk = self.ctx.kx_fresh_name("ac_block")
        idx = self.ctx.kx_fresh_name("ac_idx")
        sink = self.ctx.kx_fresh_name("ac_sink")
        code = (
            f"{{ // Cooperative shared memory load\n"
            f"    __shared__ unsigned int {buf}[256];\n"
            f"    namespace cg = cooperative_groups;\n"
            f"    cg::thread_block {blk} = cg::this_thread_block();\n"
            f"    unsigned int {idx} = {blk}.thread_rank() % 256;\n"
            f"    {buf}[{idx}] = input[(blockIdx.x * 256 + {idx}) % n];\n"
            f"    cg::sync({blk});\n"
            f"{self._kx_keep_alive(sink, f'{buf}[({idx} ^ 1) % 256]')}"
            f"}}"
        )
        return RawCode(text=code)

    def _scoped_atomic(self) -> Stmt:
        """Block-scoped atomic reduction via shared memory."""
        # Contains __syncthreads() -- must not be in divergent control flow.
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("scoped_atomic")
        bsum = self.ctx.kx_fresh_name("block_sum")
        sink = self.ctx.kx_fresh_name("sa_sink")
        src_val = self._fetch_existing_uint_name()
        # sm_60+ CUDA supports explicit block-scoped atomics.
        if self._kx_cuda_supported(60):
            variant = self.ctx.rnd_choice(["implicit", "block"])
        else:
            variant = "implicit"
        if variant == "block":
            atomic_op = f"atomicAdd_block(&{bsum}, {src_val})"
            comment = "Block-scoped atomic reduction (explicit block kx_scope)"
        else:
            atomic_op = f"atomicAdd(&{bsum}, {src_val})"
            comment = "Block-scoped atomic reduction"
        code = (
            f"{{ // {comment}\n"
            f"    __shared__ unsigned int {bsum};\n"
            f"    if (threadIdx.x == 0) {bsum} = 0;\n"
            f"    __syncthreads();\n"
            f"    {atomic_op};\n"
            f"    __syncthreads();\n"
            f"{self._kx_keep_alive(sink, bsum)}"
            f"}}"
        )
        return RawCode(text=code)

    def _coop_subwarp(self) -> Stmt:
        """Cooperative groups with sub-warp tiling (4/8/16 threads)."""
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("coop_groups")
        self.ctx._needs_coop_groups = True
        tile_size = self.ctx.rnd_choice([4, 8, 16])
        blk = self.ctx.kx_fresh_name("csw_block")
        tile = self.ctx.kx_fresh_name("csw_tile")
        lane = self.ctx.kx_fresh_name("csw_lane")
        val = self.ctx.kx_fresh_name("csw_val")
        ivar = self.ctx.kx_fresh_name("csw_i")
        fv = self.ctx.kx_fresh_name("csw_fv")
        src_val = self._fetch_existing_uint_name()
        code = (
            f"{{ // Sub-warp cooperative groups (tiled partition)\n"
            f"    namespace cg = cooperative_groups;\n"
            f"    cg::thread_block {blk} = cg::this_thread_block();\n"
            f"    cg::thread_block_tile<{tile_size}> {tile} = cg::tiled_partition<{tile_size}>({blk});\n"
            f"    unsigned int {lane} = {tile}.thread_rank();\n"
            f"    unsigned int {val} = {src_val};\n"
            f"    for (int {ivar} = {tile}.size() / 2; {ivar} > 0; {ivar} /= 2) {{\n"
            f"        {val} ^= {tile}.shfl_xor({val}, {ivar});\n"
            f"    }}\n"
            f"{self._kx_keep_alive(fv, val)}"
            f"}}"
        )
        return RawCode(text=code)

    def _coalesced_threads(self) -> Stmt:
        """cg::coalesced_threads() fan-in reduction via shfl_down."""
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        # Requires sm_70+ on CUDA; HIP supports coalesced_threads generically.
        if self._kx_is_cuda() and not self._kx_cuda_supported(70):
            return self._vardecl()
        self.ctx.features_used.add("coop_groups")
        self.ctx._needs_coop_groups = True
        active = self.ctx.kx_fresh_name("ct_active")
        val = self.ctx.kx_fresh_name("ct_val")
        ivar = self.ctx.kx_fresh_name("ct_i")
        sink = self.ctx.kx_fresh_name("ct_sink")
        src_val = self._fetch_existing_uint_name()
        code = (
            f"{{ // Coalesced threads pattern\n"
            f"    namespace cg = cooperative_groups;\n"
            f"    cg::coalesced_group {active} = cg::coalesced_threads();\n"
            f"    unsigned int {val} = {src_val};\n"
            f"    for (unsigned int {ivar} = {active}.size() / 2; {ivar} > 0; {ivar} /= 2) {{\n"
            f"        {val} += {active}.shfl_down({val}, {ivar});\n"
            f"    }}\n"
            f"{self._kx_keep_alive(sink, val)}"
            f"}}"
        )
        return RawCode(text=code)

    def _gated_barrier(self) -> Stmt:
        """Barrier behind an opaque uniform gate (DCE / barrier-placement
        stress): the condition is always true at runtime but not provable."""
        if not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("sync")
        gate = self.ctx.kx_fresh_name("gb_gate")
        val = self.ctx.kx_fresh_name("gb_val")
        buf = self.ctx.kx_fresh_name("gb_buf")
        src_val = self._fetch_existing_uint_name()
        sink = self.ctx.kx_fresh_name("gb_sink")
        code = (
            f"{{ // Gated barrier (uniform gate variable)\n"
            f"{self._kx_opaque_true(gate)}"
            f"    unsigned int {val} = {src_val} * 0x9E3779B9u;\n"
            f"    if ({gate}) {{\n"
            f"        __shared__ unsigned int {buf}[256];\n"
            f"        {buf}[threadIdx.x % 256u] = {val};\n"
            f"        __syncthreads();\n"
            f"        {val} ^= {buf}[(threadIdx.x + 1u) % 256u];\n"
            f"        __syncthreads();\n"
            f"    }}\n"
            f"{self._kx_keep_alive(sink, val)}"
            f"}}"
        )
        return RawCode(text=code)

    def _cuda_barrier(self) -> Stmt:
        """cuda::barrier arrive/wait producer/consumer handshake (sm_80+)."""
        if not self._kx_cuda_supported(80) or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("cuda_barrier")
        self.ctx._needs_cuda_barrier = True
        data = self.ctx.kx_fresh_name("cb_data")
        bar = self.ctx.kx_fresh_name("cb_bar")
        tok = self.ctx.kx_fresh_name("cb_tok")
        src_val = self._fetch_existing_uint_name()
        sink = self.ctx.kx_fresh_name("cb_sink")
        code = (
            f"{{ // cuda::barrier arrive/wait pattern (sm_80+)\n"
            f"    __shared__ unsigned int {data}[256];\n"
            f"    __shared__ cuda::barrier<cuda::thread_scope_block> {bar};\n"
            f"    if (threadIdx.x == 0) init(&{bar}, blockDim.x);\n"
            f"    __syncthreads();\n"
            f"    // Producer: write data and arrive\n"
            f"    {data}[threadIdx.x % 256u] = {src_val};\n"
            f"    auto {tok} = {bar}.arrive();\n"
            f"    // Consumer: wait and read\n"
            f"    {bar}.wait(std::move({tok}));\n"
            f"{self._kx_keep_alive(sink, f'{data}[(threadIdx.x + 1u) % 256u]')}"
            f"}}"
        )
        return RawCode(text=code)

    def _pipeline(self) -> Stmt:
        """Two-stage software pipeline with double-buffered shared memory."""
        # Contains __syncthreads() inside a loop -- safe context required.
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("pipeline")
        buf = self.ctx.kx_fresh_name("pipe_buf")
        stage = self.ctx.kx_fresh_name("pipe_stage")
        pvar = self.ctx.kx_fresh_name("pipe_p")
        nxt = self.ctx.kx_fresh_name("pipe_next")
        pval = self.ctx.kx_fresh_name("pipe_val")
        sink = self.ctx.kx_fresh_name("pipe_sink")
        code = (
            f"{{ // Two-stage pipeline pattern\n"
            f"    __shared__ unsigned int {buf}[2][256];\n"
            f"    int {stage} = 0;\n"
            f"    {buf}[0][threadIdx.x % 256] = input[(tid + 0) % n];\n"
            f"    __syncthreads();\n"
            f"    unsigned int {sink} = 0;\n"
            f"    for (int {pvar} = 1; {pvar} < 4; {pvar}++) {{\n"
            f"        int {nxt} = ({stage} + 1) & 1;\n"
            f"        {buf}[{nxt}][threadIdx.x % 256] = input[(tid + {pvar} * 256) % n];\n"
            f"        unsigned int {pval} = {buf}[{stage}][threadIdx.x % 256];\n"
            f"        {sink} ^= {pval};\n"
            f"        {stage} = {nxt};\n"
            f"        __syncthreads();\n"
            f"    }}\n"
            f"{self._kx_keep_alive(self.ctx.kx_fresh_name('pipe_ka'), sink)}"
            f"}}"
        )
        return RawCode(text=code)

    def _warp_aware(self) -> Stmt:
        """Warp-size-aware shuffle reduction (portable CUDA/HIP spelling)."""
        if not self._kx_risky_patterns_enabled() or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("warp_aware")
        val = self.ctx.kx_fresh_name("ws_val")
        offset = self.ctx.kx_fresh_name("ws_off")
        sink = self.ctx.kx_fresh_name("ws_sink")
        src_val = self._fetch_existing_uint_name()
        # CUDA removed the maskless __shfl_down; use the full-mask sync
        # variant (converged warp guaranteed by _kx_in_safe_context).  HIP
        # keeps the maskless form, its preferred spelling.
        if self._arch().kx_is_nvidia:
            shfl = f"__shfl_down_sync(0xffffffffu, {val}, {offset})"
        else:
            shfl = f"__shfl_down({val}, {offset})"
        code = (
            f"{{ // Warp-size-aware reduction\n"
            f"    unsigned int {val} = {src_val};\n"
            f"    for (unsigned int {offset} = warpSize / 2; {offset} > 0; {offset} /= 2) {{\n"
            f"        {val} += {shfl};\n"
            f"    }}\n"
            f"{self._kx_keep_alive(sink, val)}"
            f"}}"
        )
        return RawCode(text=code)

    def _cuda_syncwarp(self) -> Stmt:
        """CUDA warp-synchronization pattern (__activemask + __syncwarp)."""
        if (not self._kx_cuda_supported(70) or not self._kx_in_safe_context()
                or self.ctx.kx_is_2d):
            return self._vardecl()
        self.ctx.features_used.add("cuda_syncwarp")
        mask_name = self.ctx.kx_fresh_name("sw_mask")
        val_name = self.ctx.kx_fresh_name("sw_val")
        lane_name = self.ctx.kx_fresh_name("sw_lane")
        sink_name = self.ctx.kx_fresh_name("sw_sink")
        code = (
            f"{{ // CUDA warp sync pattern\n"
            f"    unsigned int {mask_name} = __activemask();\n"
            f"    unsigned int {lane_name} = threadIdx.x & (warpSize - 1);\n"
            f"    unsigned int {val_name} = input[tid % n];\n"
            f"    __syncwarp({mask_name});\n"
            f"{self._kx_keep_alive(sink_name, f'{val_name} ^ {mask_name}')}"
            f"}}"
        )
        return RawCode(text=code)

    def _cuda_match(self) -> Stmt:
        """__match_any_sync warp-match folded into a live uint variable."""
        if not self._kx_cuda_supported(70) or not self._kx_in_safe_context():
            return self._vardecl()
        self.ctx.features_used.add("cuda_match")
        value_type = self.ctx.rnd_choice([T.u32, T.i32, T.f32])
        value = self.expr_gen.kx_generate(value_type, depth=1)
        match = Call(T.u32, fn="__match_any_sync",
                     args=[Lit(T.u32, value=0xFFFFFFFF), value],
                     result_ty=T.u32)
        target = self._pick_target(T.u32)
        if target is None:
            vname = self.ctx.kx_fresh_name("wm")
            self.ctx.kx_declare(Var(T.u32, name=vname), divergent=True)
            return Declare(name=vname, gtype=T.u32, init=match)
        mix = BinOp(T.u32, op=Arith.BXOR, lhs=Var(T.u32, name=target.name),
                    rhs=match)
        return Assign(Var(T.u32, name=target.name), mix)

    def _cuda_cache_store(self) -> Stmt:
        """CUDA cache-hint store intrinsics (__stcg/__stcs/__stwb/__stwt)."""
        if (not self._kx_cuda_supported(70) or not self._kx_in_safe_context()
                or self.ctx.kx_is_2d):
            return self._vardecl()
        self.ctx.features_used.add("cuda_cache_store")
        if (self.ctx.kx_scope.lookup("output") is None
                or self.ctx.kx_scope.lookup("tid") is None):
            return self._vardecl()
        tid_ref = Var(T.u32, name="tid")
        n_in_scope = self.ctx.kx_scope.lookup("n") is not None
        if n_in_scope:
            n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
            idx = SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid_ref, rhs=n_u)
        else:
            idx = tid_ref
        addr = AddrOf(TPointer(T.u32),
                      arg=Subscript(T.u32,
                                    array=Var(TPointer(T.u32), name="output"),
                                    index=idx))
        value = self.expr_gen.kx_generate(T.u32, depth=1)
        fn = self.ctx.rnd_choice(["__stcg", "__stcs", "__stwb", "__stwt"])
        store = ExprStmt(Call(T.void, fn=fn, args=[addr, value],
                              result_ty=T.void))
        # Guard with ``tid < n`` like _atomic / _cuda_pipeline: with
        # vary_launch the grid can exceed n_elements (block size not evenly
        # dividing n), so threads with tid >= n would otherwise wrap via
        # ``tid % n`` onto another thread's output slot -> write race that
        # corrupts the determinism oracle.
        if n_in_scope:
            return If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_ref, rhs=n_u),
                      then=Block([store]))
        return store

    def _cuda_pipeline(self) -> Stmt:
        """CUDA pipeline-commit / wait_prior pattern (sm_80+)."""
        if (not self._kx_cuda_supported(80) or not self._kx_in_safe_context()
                or self.ctx.kx_is_2d):
            return self._vardecl()
        self.ctx.features_used.add("cuda_pipeline")
        for req in ("output", "input", "tid", "n"):
            if self.ctx.kx_scope.lookup(req) is None:
                return self._vardecl()
        buf = self.ctx.kx_fresh_name("pipe")
        lane = self.ctx.kx_fresh_name("pipe_lane")
        val = self.ctx.kx_fresh_name("pipe_val")
        mix = self.ctx.kx_fresh_name("pipe_mix")
        buf_t = TArray(T.u32, 32)
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        tid_ref = Var(T.u32, name="tid")
        idx = SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid_ref, rhs=n_u)
        lane_expr = BinOp(T.u32, op=Arith.BAND,
                          lhs=ThreadBuiltin(T.u32, axis="x", source="threadIdx"),
                          rhs=Lit(T.u32, value=31))
        buf_var = Var(buf_t, name=buf)
        lane_var = Var(T.u32, name=lane)
        val_var = Var(T.u32, name=val)
        mix_var = Var(T.u32, name=mix)
        mix_idx = BinOp(T.u32, op=Arith.BAND,
                        lhs=BinOp(T.u32, op=Arith.BXOR, lhs=lane_var,
                                  rhs=Lit(T.u32, value=1)),
                        rhs=Lit(T.u32, value=31))
        out_slot = Subscript(T.u32,
                             array=Var(TPointer(T.u32), name="output"),
                             index=tid_ref)
        stmts: List[Stmt] = [
            Declare(name=buf, gtype=buf_t, space=MemSpace.SHARED,
                    quals=[Qual.SHARED_MEM]),
            Declare(name=lane, gtype=T.u32, init=lane_expr),
            Declare(name=val, gtype=T.u32,
                    init=Subscript(T.u32,
                                   array=Var(TPointer(T.u32), name="input"),
                                   index=idx)),
            Assign(Subscript(T.u32, array=buf_var, index=lane_var), val_var),
            ExprStmt(Call(T.void, fn="__pipeline_commit", args=[],
                          result_ty=T.void)),
            ExprStmt(Call(T.void, fn="__pipeline_wait_prior",
                          args=[Lit(T.u32, value=0)], result_ty=T.void)),
            Declare(name=mix, gtype=T.u32,
                    init=Subscript(T.u32, array=buf_var, index=mix_idx)),
            If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_ref, rhs=n_u),
               then=Block([
                   Assign(out_slot,
                          BinOp(T.u32, op=Arith.BXOR, lhs=out_slot,
                                rhs=mix_var))])),
        ]
        return stmts

    # ------------------------------------------------------------------
    # template fragment (Fuzzilli-style fusion)
    # ------------------------------------------------------------------

    def _template_fragment(self):
        """Splice a 1D template's body as a composable fragment at the
        current (top-level, safe) program point.

        Returns a LIST of function-kx_scope statements (build_stmts
        flattens lists), so the template shares the kernel's flat kx_scope
        exactly like the prologue path.  Restricted to 1D tid-output
        templates and guarded against recursion; falls back to a vardecl
        when unsafe or empty.
        """
        if (not self._kx_in_safe_context() or self.ctx.kx_is_2d
                or getattr(self.ctx, "_in_template_fragment", False)):
            return self._vardecl()
        reg = getattr(self.ctx, "_fragment_registry", None)
        if reg is None:
            from generator_ir.templates.catalog import kx_build_default_registry
            reg = kx_build_default_registry()
            self.ctx._fragment_registry = reg
        candidates = [t for t in reg.kx_all_templates()
                      if not t.kx_is_2d() and not t.uses_non_tid_output()]
        if not candidates:
            return self._vardecl()
        template = self.ctx.rnd_choice(candidates)
        self.ctx._in_template_fragment = True
        try:
            body = list(template.build_body(self.ctx))
        except Exception:
            body = []
        finally:
            self.ctx._in_template_fragment = False
        if not body:
            return self._vardecl()
        self.ctx.features_used.add("template_fragment:" + template.name)
        return body

    # ------------------------------------------------------------------
    # observable_fold divergence patterns
    # ------------------------------------------------------------------

    def _divergent_switch(self, depth: int) -> Stmt:
        """switch (tid & m) with per-arm hash arithmetic on a local acc."""
        acc_t = T.u32
        acc_name = self.ctx.kx_fresh_name("dsw")
        tid = Var(T.u32, name="tid")
        seed = BinOp(T.u32, op=Arith.BXOR, lhs=tid,
                     rhs=Lit(T.u32, value=0xDEADBEEF))
        decl = Declare(name=acc_name, gtype=acc_t, init=seed)
        self.ctx.kx_declare(Var(acc_t, name=acc_name))

        def rotl(x: Expr, k: int) -> Expr:
            return BinOp(T.u32, op=Arith.BOR,
                         lhs=SafeBin(T.u32, op=SafeOp.SSHL, lhs=x, rhs=Lit(T.u32, value=k)),
                         rhs=SafeBin(T.u32, op=SafeOp.SSHR, lhs=x,
                                     rhs=Lit(T.u32, value=32 - k)))

        def acc_ref() -> Var:
            return Var(acc_t, name=acc_name)

        arms = [
            BinOp(T.u32, op=Arith.BXOR,
                  lhs=BinOp(T.u32, op=Arith.MUL, lhs=acc_ref(), rhs=Lit(T.u32, value=0x9E3779B9)),
                  rhs=SafeBin(T.u32, op=SafeOp.SSHR, lhs=acc_ref(), rhs=Lit(T.u32, value=16))),
            BinOp(T.u32, op=Arith.MUL,
                  lhs=BinOp(T.u32, op=Arith.BXOR, lhs=acc_ref(), rhs=Lit(T.u32, value=0x85EBCA6B)),
                  rhs=Lit(T.u32, value=0xC2B2AE35)),
            BinOp(T.u32, op=Arith.ADD, lhs=acc_ref(),
                  rhs=BinOp(T.u32, op=Arith.BXOR,
                            lhs=SafeBin(T.u32, op=SafeOp.SSHL, lhs=acc_ref(), rhs=Lit(T.u32, value=13)),
                            rhs=SafeBin(T.u32, op=SafeOp.SSHR, lhs=acc_ref(), rhs=Lit(T.u32, value=7)))),
            BinOp(T.u32, op=Arith.BXOR, lhs=rotl(acc_ref(), 7), rhs=Lit(T.u32, value=0x27D4EB2F)),
        ]
        cases = [Case(value=Lit(T.i32, value=i),
                      body=Block([Assign(acc_ref(),
                                         BinOp(T.u32, op=Arith.BXOR,
                                               lhs=acc_ref(),
                                               rhs=arm))]))
                 for i, arm in enumerate(arms)]
        cases.append(Case(value=None, body=Block([
            Assign(acc_ref(),
                   BinOp(T.u32, op=Arith.ADD, lhs=acc_ref(),
                         rhs=Lit(T.u32, value=0x517CC1B7)))])))
        sw = Switch(expr=BinOp(T.u32, op=Arith.BAND, lhs=tid,
                               rhs=Lit(T.u32, value=len(arms) - 1)),
                    cases=cases)
        # Fold acc into the observed checksum if we are in weave/observable mode.
        # Return a FLAT list (no Block wrapper): the accumulator declaration is
        # registered in the current kx_scope, so the statements must live in the
        # enclosing block too -- wrapping them in a Block would let later
        # statements reference a name that is out of kx_scope in C.
        fold = self._fold_stmt(acc_ref())
        out: List[Stmt] = [decl, sw]
        if fold is not None:
            out.append(fold)
        return out

    def _strided_unroll(self, depth: int) -> Stmt:
        """for (j < U) acc ^= input[(tid % n) + j*n]  (stride-n reads)."""
        n_u = Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n"))
        tid = Var(T.u32, name="tid")
        u_max = max(2, int(getattr(self.ctx.config, "strided_unroll_depth", 32)))
        unroll = min(self.ctx.rnd_choice([8, 16, 32]), u_max)
        acc_name = self.ctx.kx_fresh_name("su")
        base = SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid, rhs=n_u)
        decl = Declare(name=acc_name, gtype=T.u32,
                       init=Subscript(T.u32,
                                      array=Var(TPointer(T.u32), name="input"),
                                      index=base))
        self.ctx.kx_declare(Var(T.u32, name=acc_name))
        j_var = "j_" + acc_name.split("_", 1)[1]
        idx = BinOp(T.u32, op=Arith.ADD, lhs=base,
                    rhs=BinOp(T.u32, op=Arith.MUL,
                              lhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name=j_var)),
                              rhs=n_u))
        body = Block([
            Assign(Var(T.u32, name=acc_name),
                   BinOp(T.u32, op=Arith.BXOR,
                         lhs=Var(T.u32, name=acc_name),
                         rhs=Subscript(T.u32,
                                       array=Var(TPointer(T.u32), name="input"),
                                       index=idx))),
        ])
        loop = For(var=j_var, start=Lit(T.i32, value=0),
                   end=Lit(T.i32, value=unroll), step=Lit(T.i32, value=1),
                   body=body)
        fold = self._fold_stmt(Var(T.u32, name=acc_name))
        out: List[Stmt] = [decl, loop]
        if fold is not None:
            out.append(fold)
        return out

    def _fold_stmt(self, value: Expr) -> Optional[Stmt]:
        """``_fv ^= (uint)value`` when _fv is part of this kernel."""
        names = self.ctx.kx_scope.declared_names()
        if "_fv" not in names:
            return None
        return Assign(Var(T.u32, name="_fv"),
                      BinOp(T.u32, op=Arith.BXOR,
                            lhs=Var(T.u32, name="_fv"),
                            rhs=Cast(T.u32, to=T.u32, arg=value)))
