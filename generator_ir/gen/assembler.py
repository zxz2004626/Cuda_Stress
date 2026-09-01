"""Kernel generator: assembles complete programs from templates + generators.

The generated kernel contract:

    __global__ void kernel_main(unsigned int* output, const unsigned int* input,
                                int n, <scalar params...>) {
        <decls>
        for (int _rep = 0; _rep < R; _rep++) {
            <template body + generated statements + _fv folds + filler>
        }
        <stress sections>
        <single observed write of the checksum into output>
    }

Determinism: every value feeding ``output`` is a deterministic function of
``input`` (host-initialized), thread geometry, and constants.  No races:
per-thread output slots, bounded reads, uniform barriers only.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from generator_ir.ir.typedecl import (
    GType, T, MemSpace, Qual, TArray, TPointer, TQualified,
    TUInt, TInt, TSChar, TChar, TShort, TUShort, TLong, TULong, TBool,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, Cast, SafeBin,
    Subscript, Call, Fma,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, ExprStmt, Declare, Block, If, For,
    Barrier, Fence, Return, RawCode, UnrollHint,
)
from generator_ir.ir.program import Func, Program, LaunchConfig
from generator_ir.gen.options import GenConfig
from generator_ir.gen.context import GenContext
from generator_ir.gen.expr_builder import ExprGen
from generator_ir.gen.stmt_builder import StmtGen
from generator_ir.templates.skeleton import KernelTemplate, GenericKernelScaffold
from generator_ir.templates.catalog import TemplateRegistry, kx_build_default_registry


class KernelGenerator:
    """Builds one random kernel Program from a GenConfig."""

    def __init__(self, config: GenConfig,
                 registry: Optional[TemplateRegistry] = None) -> None:
        self.config = config
        # Keep the registry lazy: a generic-baseline run must not even select
        # or instantiate the targeted template catalog.
        self.registry = registry

    def _get_registry(self) -> TemplateRegistry:
        if self.registry is None:
            self.registry = kx_build_default_registry()
        return self.registry

    # ------------------------------------------------------------------
    # top-level
    # ------------------------------------------------------------------

    def kx_generate(self, template_name: Optional[str] = None) -> Program:
        ctx = GenContext(self.config)

        if getattr(self.config, "generic_baseline", False):
            if template_name is not None:
                raise ValueError("generic_baseline does not accept a template")
            template = GenericKernelScaffold(self.config.element_count)
            ctx.features_used.add("generic_baseline")
        elif template_name is not None:
            registry = self._get_registry()
            template = registry.get(template_name)
            if template is None:
                raise ValueError(f"unknown template {template_name!r}; "
                                 f"have {registry.kx_all_names()}")
        else:
            template = self._get_registry().kx_weighted_random(
                ctx.rng, schedule=ctx.weights)

        ctx.kx_is_2d = template.kx_is_2d() or template.uses_non_tid_output()

        block = template.preferred_block()
        grid = template.preferred_grid()
        if self.config.threads_per_block_dim != (256, 1, 1):
            block = self.config.threads_per_block_dim
        if self.config.blocks_per_grid_dim != (128, 1, 1):
            grid = self.config.blocks_per_grid_dim

        # Launch diversity: randomize the 1D block size (grid rescaled so
        # total threads never drop below the template's coverage).  Skip
        # shared-memory templates: they size scratch to preferred_block() and
        # bind reduction/scan logic to it, so a block >256 indexes shared
        # memory out of bounds and a block <256 reads uninitialised slots
        # (cf. weaver._weave_uses_fragile_256_shape, which pins these shapes).
        if (getattr(self.config, "vary_launch", False)
                and not template.kx_is_2d()
                and template.kx_required_shared_mem() == 0):
            total = block[0] * grid[0]
            new_bx = ctx.rnd_choice([64, 96, 128, 192, 256, 512])
            new_gx = max(1, -(-total // new_bx))
            block = (new_bx, 1, 1)
            grid = (new_gx, 1, 1)

        repeats = self.config.repeat_rounds()
        self._randomize_stress(ctx)

        launch = LaunchConfig(
            grid=grid,
            block=block,
            shared_bytes=template.kx_required_shared_mem(),
            body_repeat=repeats,
            n_elements=template.kx_required_n_elements(),
            kx_is_2d=template.kx_is_2d(),
            observable_fold=self.config.observable_fold,
            complexity=self.config.complexity,
            input_size_mult=(self.config.strided_unroll_depth
                             if (self.config.observable_fold and not ctx.kx_is_2d
                                 and not getattr(self.config, "generic_baseline", False))
                             else 1),
            arch=self.config.isa,
            allow_volatile=self.config.allow_volatile,
            input_dist=getattr(self.config, "input_dist", "hash"),
            codegen=self.config.codegen,
            generic_baseline=getattr(self.config, "generic_baseline", False),
            checksum_only=self.config.checksum_only,
        )

        if ctx.config.use_full_int_widths:
            arr_decls, arr_info = self._kx_build_device_arrays(ctx)
        else:
            arr_decls, arr_info = [], []
        ctx._device_arrays = arr_info

        helpers = self._build_helpers(ctx)
        kernel = self._kx_build_kernel(ctx, template, repeats, helpers)

        # __launch_bounds__ must cover the actual block size.
        actual = block[0] * block[1] * block[2]
        choices = [t for t in (256, 384, 512, 768, 1024) if t >= actual] or [1024]
        if not getattr(self.config, "generic_baseline", False):
            kernel.launch_bounds = (ctx.rnd_choice(choices), ctx.rnd_choice([1, 2]))

        if (ctx.config.target == "cuda"
                and not getattr(self.config, "generic_baseline", False)):
            if (ctx.kx_arch_number() >= 90 and grid[0] % 2 == 0
                    and actual <= 512 and ctx.rnd_bool(0.35)):
                kernel.cluster_dims = (2, 1, 1)

        if getattr(ctx, "_needs_dynamic_shared", False) and launch.shared_bytes == 0:
            launch.shared_bytes = max(1024, actual * 4)
        if getattr(ctx, "_needs_coop_groups", False):
            launch.extra_headers.add("<cooperative_groups.h>")
        if getattr(ctx, "_needs_coop_groups_memcpy_async", False):
            launch.extra_headers.add("<cooperative_groups/memcpy_async.h>")
        if getattr(ctx, "_needs_cuda_barrier", False):
            launch.extra_headers.add("<cuda/barrier>")
        if getattr(ctx, "_needs_constant_mem", False):
            launch.needs_constant_mem = True
        launch.generation_features = set(ctx.features_used)

        program = Program(launch)
        for d in arr_decls:
            program.add(d)
        for h in helpers:
            program.add(h)
        program.add(kernel)
        return program

    def _randomize_stress(self, ctx: GenContext) -> None:
        """Draw post-loop stress sections per complexity."""
        c = self.config
        if (getattr(c, "generic_baseline", False)
                or getattr(c, "pass_targeted_only", False)):
            c.do_bank_stress = False
            c.reg_pressure_level = "none"
            c.do_vector_stress = False
            c.do_switch_stress = False
            c.do_barrier_stress = False
            return
        if c.complexity == "medium":
            c.do_bank_stress = ctx.rnd_bool(0.40)
            c.reg_pressure_level = ctx.rnd_choice(
                ["none", "none", "none", "low", "low"])
            c.do_vector_stress = ctx.rnd_bool(0.40)
            c.do_switch_stress = ctx.rnd_bool(0.40)
            c.do_barrier_stress = ctx.rnd_bool(0.40)
        elif c.complexity == "high":
            c.do_bank_stress = ctx.rnd_bool(0.60)
            c.reg_pressure_level = ctx.rnd_choice(
                ["none", "low", "low", "medium", "medium"])
            c.do_vector_stress = ctx.rnd_bool(0.60)
            c.do_switch_stress = ctx.rnd_bool(0.60)
            c.do_barrier_stress = ctx.rnd_bool(0.60)
        elif c.complexity == "stress":
            c.do_bank_stress = True
            c.reg_pressure_level = ctx.rnd_choice(
                ["medium", "high", "high", "extreme"])
            c.do_vector_stress = True
            c.do_switch_stress = True
            c.do_barrier_stress = True

    # ------------------------------------------------------------------
    # device arrays / helpers
    # ------------------------------------------------------------------

    _DEVICE_ARRAY_TYPES = [
        T.schar, T.uchar, T.short, T.ushort, T.i32, T.u32, T.i64, T.u64, T.boolean,
    ]

    def _kx_build_device_arrays(self, ctx: GenContext) -> Tuple[List[Declare], List[dict]]:
        """Random __device__ global arrays (zero-initialized by the harness)."""
        if ctx.config.complexity == "stress":
            count = ctx.rnd_int(10, 18)
        else:
            count = ctx.rnd_int(3, 9)
        decls: List[Declare] = []
        info: List[dict] = []
        for i in range(count):
            elem = ctx.rnd_choice(self._DEVICE_ARRAY_TYPES)
            if ctx.config.complexity == "stress":
                ndim = ctx.rnd_weighted([("1d", 10), ("2d", 40), ("3d", 50)])
            else:
                ndim = ctx.rnd_weighted([("1d", 30), ("2d", 40), ("3d", 30)])
            size = ctx.rnd_choice([7, 11, 13, 16])
            at: GType = TArray(elem, size)
            dims = [size]
            if ndim in ("2d", "3d"):
                at = TArray(at, size)
                dims.append(size)
            if ndim == "3d":
                at = TArray(at, size)
                dims.append(size)
            name = f"arr_{i}"
            decls.append(Declare(name=name, gtype=at,
                                 space=MemSpace.GLOBAL, quals=[Qual.DEVICE]))
            info.append({"name": name, "type": at, "dims": dims, "elem": elem})
        return decls, info

    def _build_helpers(self, ctx: GenContext,
                       weave: bool = False) -> List[Func]:
        """Random __device__ helper functions called from the kernel body."""
        stress = ctx.config.complexity == "stress"
        count = ctx.rnd_int(4, 8) if stress else ctx.rnd_int(2, 4)
        helpers: List[Func] = []

        for i in range(count):
            n_params = ctx.rnd_int(3, 5) if stress else ctx.rnd_int(2, 4)
            params = []
            for j in range(n_params):
                ptype = ctx.rnd_choice([T.u32, T.u32, T.u32, T.i32])
                params.append(Declare(name=f"p{j}", gtype=ptype))

            helper_ctx = GenContext(self.config)
            helper_ctx.rng = ctx.rng           # continue the same stream
            helper_ctx.in_device_helper = True  # no barriers inside helpers
            if weave:
                helper_ctx._weave_mode = True
            for p in params:
                helper_ctx.kx_declare(Var(p.gtype, name=p.name))

            expr_gen = ExprGen(helper_ctx)
            stmt_gen = StmtGen(helper_ctx, expr_gen)
            body_count = ctx.rnd_int(10, 25) if stress else ctx.rnd_int(5, 15)
            body_stmts = stmt_gen.build_stmts(body_count)

            ret_expr = expr_gen.kx_generate(T.u32)
            if weave:
                # Fold params + locals into the return so the body is not dead.
                acc = ret_expr
                for v in helper_ctx.kx_function_scope_variables():
                    if not (isinstance(v.ty, GType) and v.ty.is_int):
                        continue
                    acc = BinOp(T.u32, op=Arith.BXOR,
                                lhs=BinOp(T.u32, op=Arith.MUL, lhs=acc,
                                          rhs=Lit(T.u32, value=0x9E3779B1)),
                                rhs=Cast(T.u32, to=T.u32,
                                         arg=Var(v.ty, name=v.name)))
                ret_expr = acc
            body_stmts.append(Return(value=ret_expr))

            inline_hint = (None if getattr(ctx.config, "generic_baseline", False)
                           else ctx.rnd_choice(
                               [None, "__forceinline__", "__noinline__", None]))
            helpers.append(Func(
                name=f"helper_{i}", params=params, ret=T.u32,
                body=Block(body_stmts), quals=[Qual.DEVICE],
                inline_hint=inline_hint,
            ))
        return helpers

    # ------------------------------------------------------------------
    # kernel body
    # ------------------------------------------------------------------

    def _kx_build_kernel(self, ctx: GenContext, template: KernelTemplate,
                      repeats: int, helpers: List[Func]) -> Func:
        u32 = T.u32

        # -- parameters ---------------------------------------------------
        out_param = Declare(name="output", gtype=TPointer(u32),
                            quals=[Qual.RESTRICT])
        in_param = Declare(name="input", gtype=TPointer(u32),
                           quals=[Qual.RESTRICT, Qual.CONST])
        n_param = Declare(name="n", gtype=T.i32)
        output_var = Var(TPointer(u32), name="output")
        input_var = Var(TPointer(u32), name="input")
        n_var = Var(T.i32, name="n")
        ctx.kx_declare(output_var)
        ctx.kx_declare(input_var)
        ctx.kx_declare(n_var)

        extra_params: List[Declare] = []
        for i in range(self.config.scalar_param_count):
            ptype = ctx.rnd_choice(ctx.declaration_types())
            pname = f"var_{i}"
            extra_params.append(Declare(name=pname, gtype=ptype))
            ctx.kx_declare(Var(ptype, name=pname))

        stmts: List[Stmt] = []

        # -- tid ------------------------------------------------------------
        if ctx.kx_is_2d:
            lin_thread = BinOp(u32, op=Arith.ADD,
                lhs=ThreadBuiltin(u32, axis="x", source="threadIdx"),
                rhs=BinOp(u32, op=Arith.MUL,
                          lhs=ThreadBuiltin(u32, axis="y", source="threadIdx"),
                          rhs=ThreadBuiltin(u32, axis="x", source="blockDim")))
            lin_block = BinOp(u32, op=Arith.ADD,
                lhs=ThreadBuiltin(u32, axis="x", source="blockIdx"),
                rhs=BinOp(u32, op=Arith.MUL,
                          lhs=ThreadBuiltin(u32, axis="y", source="blockIdx"),
                          rhs=ThreadBuiltin(u32, axis="x", source="gridDim")))
            per_block = BinOp(u32, op=Arith.MUL,
                lhs=ThreadBuiltin(u32, axis="x", source="blockDim"),
                rhs=ThreadBuiltin(u32, axis="y", source="blockDim"))
            tid_expr = BinOp(u32, op=Arith.ADD,
                             lhs=BinOp(u32, op=Arith.MUL, lhs=lin_block, rhs=per_block),
                             rhs=lin_thread)
        else:
            tid_expr = BinOp(u32, op=Arith.ADD,
                lhs=ThreadBuiltin(u32, axis="x", source="threadIdx"),
                rhs=BinOp(u32, op=Arith.MUL,
                          lhs=ThreadBuiltin(u32, axis="x", source="blockIdx"),
                          rhs=ThreadBuiltin(u32, axis="x", source="blockDim")))
        stmts.append(Declare(name="tid", gtype=u32, init=tid_expr))
        tid_var = Var(u32, name="tid")
        ctx.kx_declare(tid_var, divergent=True)

        # Optional compiler memory barrier (L2 interaction probe).
        if (not getattr(self.config, "generic_baseline", False)
                and not getattr(self.config, "pass_targeted_only", False)
                and ctx.rnd_bool(0.10)):
            ctx.features_used.add("l2_cache_barrier")
            stmts.append(RawCode(text='asm volatile("" ::: "memory");'))

        if helpers:
            ctx._device_helpers = helpers

        # -- bulk pre-loop declarations ------------------------------------
        bulk_expr = ExprGen(ctx)
        # Account for the outer _rep loop in the trip budget so nested loops
        # don't explode to millions of iterations (62*44*23*4 = ~700k).
        ctx.loop_trip_product = max(1, repeats)
        pre_decls: List[Stmt] = []
        num_bulk = max(0, self.config.stmt_budget - ctx.stmt_count)
        for _ in range(num_bulk):
            vt = ctx.rnd_choice(ctx.declaration_types())
            vn = ctx.kx_fresh_name("v")
            pre_decls.append(Declare(name=vn, gtype=vt,
                                     init=bulk_expr.kx_generate(vt, depth=0)))
            ctx.kx_declare(Var(vt, name=vn))

        bulk_stmt_gen = StmtGen(ctx, bulk_expr)
        # Hoist scalar Declares into the pre-loop block.  A Declare may only
        # hoist when every name its initializer references was declared before
        # this batch or by an earlier hoisted Declare -- otherwise it could be
        # separated from a shared-array declaration or a RawCode-declared name
        # that stays inside the loop body (use-before-def).
        from generator_ir.ir.traversal import collect as collect_nodes
        hoistable = set(ctx.kx_scope.declared_names())
        extra = bulk_stmt_gen.build_stmts(min(10, num_bulk // 3))
        loop_body_extra: List[Stmt] = []
        for s in extra:
            if isinstance(s, Declare) and not isinstance(s.gtype, TArray):
                refs = ({n.name for n in collect_nodes(
                    s.init, lambda n: isinstance(n, Var))}
                    if s.init is not None else set())
                if refs <= hoistable:
                    pre_decls.append(s)
                    hoistable.add(s.name)
                    continue
            loop_body_extra.append(s)

        # -- template body ---------------------------------------------------
        saved_visible = ctx._visible_names
        ctx._visible_names = set(hoistable)
        if (not getattr(self.config, "generic_baseline", False)
                and self.config.observable_fold and not ctx.kx_is_2d
                and ctx.rnd_bool(0.25)):
            template_body: List[Stmt] = []
            ctx.features_used.add("no_prologue_template")
        else:
            try:
                template_body = list(template.build_body(ctx))
            finally:
                ctx._visible_names = saved_visible
        ctx._visible_names = saved_visible
        template_body.extend(loop_body_extra)

        n_u = Cast(u32, to=u32, arg=n_var)

        # -- persistent output vars (loop-carried state) ---------------------
        output_vars: List[Var] = []
        n_ov = 20 if ctx.config.complexity == "stress" else 8
        for oi in range(n_ov):
            ovt = ctx.rnd_choice(ctx.declaration_types())
            ovn = ctx.kx_fresh_name("ov")
            init = Cast(ovt, to=ovt,
                        arg=Subscript(u32, array=input_var,
                                      index=SafeBin(u32, op=SafeOp.SMOD,
                                                    lhs=BinOp(u32, op=Arith.ADD,
                                                              lhs=tid_var,
                                                              rhs=Lit(u32, value=oi)),
                                                    rhs=n_u)))
            stmts.append(Declare(name=ovn, gtype=ovt, init=init))
            ov = Var(ovt, name=ovn)
            ctx.kx_declare(ov)
            output_vars.append(ov)

        stmts.extend(pre_decls)

        # -- compute filler (pre/loop/post) ----------------------------------
        fv_pre, fv_loop, fv_post = self._build_filler(ctx, input_var, tid_var,
                                                      repeats)
        stmts.extend(fv_pre)

        # Loop-carried XOR mixing for output vars (ov ^= ov*C + _fv style).
        safe_sink_types = (TUInt, TInt, TSChar, TChar, TShort, TUShort,
                           TLong, TULong)
        mix_stmts: List[Stmt] = []
        for sv in [v for v in output_vars if isinstance(v.ty, safe_sink_types)]:
            ref = Var(sv.ty, name=sv.name)
            mixed = BinOp(u32, op=Arith.BXOR,
                          lhs=Cast(u32, to=u32, arg=ref),
                          rhs=BinOp(u32, op=Arith.ADD,
                                    lhs=BinOp(u32, op=Arith.MUL,
                                              lhs=Cast(u32, to=u32,
                                                       arg=Var(T.i32, name="_rep")),
                                              rhs=Lit(u32, value=0x85EBCA6B)),
                                    rhs=Var(u32, name="_fv")))
            mix_stmts.append(Assign(ref, Cast(sv.ty, to=sv.ty, arg=mixed)))

        # Checksum folds: _fv ^= (uint)v for a random subset of kx_scope vars.
        fold_types = ((TUInt, TUShort, TULong, TBool)
                      if self.config.observable_fold else safe_sink_types)
        foldable = [v for v in ctx.kx_function_scope_variables()
                    if v.name not in ("_fv", "_rep", "_ci", "tid", "output",
                                      "input", "n")
                    and isinstance(v.ty, fold_types)]
        if foldable:
            keep_p = ctx.rng.uniform(0.4, 1.0)
            folded = [v for v in foldable if ctx.rng.random() < keep_p]
            min_keep = min(4, len(foldable))
            if len(folded) < min_keep:
                folded = foldable[:min_keep]
        else:
            folded = []
        checksum_stmts: List[Stmt] = []
        for v in folded:
            checksum_stmts.append(Assign(
                Var(u32, name="_fv"),
                BinOp(u32, op=Arith.BXOR,
                      lhs=Var(u32, name="_fv"),
                      rhs=Cast(u32, to=u32, arg=Var(v.ty, name=v.name)))))

        # Per-iteration output reset (1D only; 2D/non-tid templates race).
        reset_stmts: List[Stmt] = []
        if not ctx.kx_is_2d:
            reset_stmts = [
                If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_var, rhs=n_u),
                   then=Block([Assign(Subscript(u32, array=output_var, index=tid_var),
                                      Lit(u32, value=0))])),
            ]
            if not getattr(self.config, "generic_baseline", False):
                reset_stmts.append(Barrier())

        # Barrier-pressure block between template and filler (1D, non-fold).
        barrier_stmts: List[Stmt] = []
        if (not getattr(self.config, "generic_baseline", False)
                and not ctx.kx_is_2d and not self.config.observable_fold
                and ctx.rnd_bool(0.35)):
            barrier_stmts = self._kx_build_barrier_pressure(
                ctx, ExprGen(ctx), output_var, tid_var, n_var)

        if self.config.observable_fold:
            if not ctx.kx_is_2d:
                half = len(loop_body_extra) // 2
                template_core = template_body[:len(template_body) - len(loop_body_extra)]
                all_body = (loop_body_extra[:half] + template_core
                            + loop_body_extra[half:] + checksum_stmts + mix_stmts)
            else:
                all_body = template_body + checksum_stmts + mix_stmts + fv_loop
        else:
            all_body = (reset_stmts + template_body + barrier_stmts
                        + checksum_stmts + fv_loop)

        stmts.append(For(var="_rep", start=Lit(T.i32, value=0),
                         end=Lit(T.i32, value=repeats), step=Lit(T.i32, value=1),
                         body=Block(all_body)))

        # -- post-loop stress sections ---------------------------------------
        stress = self._build_stress_sections(ctx, tid_var)
        stmts.extend(stress)

        # -- final write ------------------------------------------------------
        if self.config.observable_fold:
            bound = (BinOp(u32, op=Arith.MUL, lhs=n_u, rhs=n_u)
                     if template.kx_is_2d() else n_u)
            stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_var, rhs=bound),
                            then=Block([Assign(
                                Subscript(u32, array=output_var, index=tid_var),
                                Var(u32, name="_fv"))])))
        else:
            stmts.extend(fv_post)
            if output_vars:
                sink: Expr = Cast(u32, to=u32,
                                  arg=Var(output_vars[0].ty, name=output_vars[0].name))
                for ov in output_vars[1:]:
                    sink = BinOp(u32, op=Arith.BXOR, lhs=sink,
                                 rhs=Cast(u32, to=u32, arg=Var(ov.ty, name=ov.name)))
                stmts.append(Declare(name="_ov_sink", gtype=u32, init=sink))
            if not ctx.kx_is_2d:
                stmts.extend(self._kx_build_checksum(output_var, tid_var, n_u))

        return Func(
            name="kernel_main",
            params=[out_param, in_param, n_param] + extra_params,
            ret=T.void,
            body=Block(stmts),
            quals=[Qual.KERNEL],
        )

    # ------------------------------------------------------------------
    # filler / barrier pressure / checksum
    # ------------------------------------------------------------------

    def _build_filler(self, ctx: GenContext, input_var: Var, tid_var: Var,
                      repeats: int):
        """_fv accumulator + inner mixing loop + post-loop sink."""
        u32 = T.u32
        n_u = Cast(u32, to=u32, arg=Var(T.i32, name="n"))
        pre = [Declare(name="_fv", gtype=u32,
                       init=Subscript(u32, array=input_var,
                                      index=SafeBin(u32, op=SafeOp.SMOD,
                                                    lhs=tid_var, rhs=n_u)))]

        if repeats < 4:
            inner_rounds = 16
        elif repeats < 100:
            inner_rounds = 32
        else:
            inner_rounds = 256

        fv = Var(u32, name="_fv")
        line1 = Assign(fv, BinOp(u32, op=Arith.BXOR,
            lhs=BinOp(u32, op=Arith.ADD,
                      lhs=BinOp(u32, op=Arith.MUL, lhs=fv,
                                rhs=Lit(u32, value=2654435761)),
                      rhs=SafeBin(u32, op=SafeOp.SSHR, lhs=fv, rhs=Lit(u32, value=17))),
            rhs=SafeBin(u32, op=SafeOp.SSHL, lhs=fv, rhs=Lit(u32, value=13))))
        line2 = Assign(fv, BinOp(u32, op=Arith.BXOR,
            lhs=fv,
            rhs=BinOp(u32, op=Arith.ADD,
                      lhs=BinOp(u32, op=Arith.ADD, lhs=fv,
                                rhs=Lit(u32, value=0x9E3779B9)),
                      rhs=tid_var)))
        inner = For(var="_ci", start=Lit(T.i32, value=0),
                    end=Lit(T.i32, value=inner_rounds),
                    step=Lit(T.i32, value=1),
                    body=Block([line1, line2]))
        post = [Declare(name="_fv_sink", gtype=u32, init=Var(u32, name="_fv"))]
        return pre, [inner], post

    def _kx_build_barrier_pressure(self, ctx: GenContext, expr_gen: ExprGen,
                                output_var: Var, tid_var: Var,
                                n_var: Var) -> List[Stmt]:
        """Shared store + live scalars before a barrier, consumed after."""
        u32 = T.u32
        n_bp = ctx.rnd_int(2, 6)
        smem_name = ctx.kx_fresh_name("bc_smem")
        # Must span the largest block (vary_launch 1D caps at 512): the
        # pre-barrier write smem[tid % size] would otherwise collide across
        # warps (threads t and t+size write one slot with distinct per-thread
        # values -> write race -> nondeterministic output[tid]).
        smem_size = ctx.rnd_choice([512, 1024, 2048])
        lane = ThreadBuiltin(u32, axis="x", source="threadIdx")

        stmts: List[Stmt] = [
            Declare(name=smem_name, gtype=TArray(u32, smem_size),
                    space=MemSpace.SHARED, quals=[Qual.SHARED_MEM]),
            Assign(Subscript(u32, array=Var(TArray(u32, smem_size), name=smem_name),
                             index=SafeBin(u32, op=SafeOp.SMOD, lhs=lane,
                                           rhs=Lit(u32, value=smem_size))),
                   expr_gen.kx_generate(u32)),
        ]
        bp_vars: List[Var] = []
        for _ in range(n_bp):
            name = ctx.kx_fresh_name("bp")
            bp_vars.append(Var(u32, name=name))
            stmts.append(Declare(name=name, gtype=u32,
                                 init=expr_gen.kx_generate(u32)))
            ctx.kx_declare(bp_vars[-1])
        stmts.append(Barrier())

        read_idx = SafeBin(u32, op=SafeOp.SMOD,
                           lhs=BinOp(u32, op=Arith.BXOR, lhs=lane, rhs=Lit(u32, value=1)),
                           rhs=Lit(u32, value=smem_size))
        accum: Expr = Subscript(u32, array=Var(TArray(u32, smem_size), name=smem_name),
                                index=read_idx)
        for v in bp_vars:
            accum = BinOp(u32, op=Arith.BXOR, lhs=accum, rhs=v)
        out_slot = Subscript(u32, array=output_var, index=tid_var)
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_var,
                                   rhs=Cast(u32, to=u32, arg=n_var)),
                        then=Block([Assign(out_slot,
                            BinOp(u32, op=Arith.BXOR, lhs=out_slot, rhs=accum))])))
        return stmts

    def _kx_build_checksum(self, output_var: Var, tid_var: Var, n_u: Expr) -> List[Stmt]:
        """Per-thread in-place hash of output[tid]; the host folds the buffer."""
        u32 = T.u32
        hv = Var(u32, name="_hash_val")
        golden = Lit(u32, value=0x9E3779B9)
        mix = BinOp(u32, op=Arith.BXOR,
            lhs=hv,
            rhs=BinOp(u32, op=Arith.ADD,
                      lhs=BinOp(u32, op=Arith.ADD,
                                lhs=BinOp(u32, op=Arith.ADD, lhs=hv, rhs=golden),
                                rhs=SafeBin(u32, op=SafeOp.SSHL, lhs=hv, rhs=Lit(u32, value=6))),
                      rhs=SafeBin(u32, op=SafeOp.SSHR, lhs=hv, rhs=Lit(u32, value=2))))
        body = Block([
            Declare(name="_hash_val", gtype=u32,
                    init=Subscript(u32, array=output_var, index=tid_var, locked=True)),
            Assign(hv, mix),
            Assign(Subscript(u32, array=output_var, index=tid_var, locked=True), hv),
        ])
        return [If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid_var, rhs=n_u),
                   then=body)]

    # ------------------------------------------------------------------
    # post-loop stress sections (verbatim C fragments)
    # ------------------------------------------------------------------

    def _build_stress_sections(self, ctx: GenContext, tid_var: Var) -> List[Stmt]:
        """Post-loop stress sections (delegates to kernel_stress module)."""
        if getattr(self.config, "generic_baseline", False):
            return []
        from generator_ir.gen.kernel_stress import (
            BANK_CONFLICT_BLOCK, REG_PRESSURE_BLOCK, VECTORIZE_BLOCK,
            SWITCH_DIVERGENCE_BLOCK, BARRIER_SPAN_BLOCK, fp_island_block,
        )
        out: List[Stmt] = []
        c = self.config
        if c.do_bank_stress:
            out.append(RawCode(text=BANK_CONFLICT_BLOCK))
        if c.reg_pressure_level != "none":
            out.append(RawCode(text=REG_PRESSURE_BLOCK(c.reg_pressure_level)))
        if c.do_vector_stress:
            out.append(RawCode(text=VECTORIZE_BLOCK))
        if c.do_switch_stress:
            out.append(RawCode(text=SWITCH_DIVERGENCE_BLOCK))
        # BARRIER_SPAN folds (unsigned long long)input -- the device pointer's
        # runtime address -- into _fv.  In observable_fold mode _fv reaches the
        # cross-run-compared output, and cudaMalloc hands the input buffer a
        # different address in the old vs new process run -> false checksum
        # mismatch.  Skip it there (matches the fp_island gate below).
        if c.do_barrier_stress and not getattr(c, "observable_fold", False):
            out.append(RawCode(text=BARRIER_SPAN_BLOCK))
        # fp_island folds FP transcendentals into _fv; in observable_fold mode
        # _fv reaches the cross-version-compared output, and FP results diverge
        # across nvcc versions -> false mismatch.  Skip it there (in non-fold
        # mode _fv is dead, so the FP fold is harmless perf stress).
        if (getattr(ctx, "use_float_types", False)
                and not getattr(c, "observable_fold", False)):
            rounds = ctx.rnd_int(4, 12)
            trans = ctx.rnd_choice(["__sinf", "__cosf", "__expf", "sqrtf", "__logf"])
            out.append(RawCode(text=fp_island_block(rounds, trans)))
        return out
def build_kernel(config: GenConfig,
                    template_name: Optional[str] = None,
                    registry: Optional[TemplateRegistry] = None) -> Program:
    """Generate a program; routes to the weaving path when config.codegen."""
    if (getattr(config, "generic_baseline", False)
            and getattr(config, "codegen", False)):
        raise ValueError("generic_baseline cannot be combined with codegen/weave mode")
    if registry is not None and getattr(config, "codegen", False):
        raise ValueError("custom template registries are not supported in codegen/weave mode")
    if getattr(config, "codegen", False):
        from generator_ir.gen.weaver import generate_weave_kernel
        return generate_weave_kernel(config, template_name)
    return KernelGenerator(config, registry=registry).kx_generate(template_name)
