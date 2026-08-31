"""Compute-stress templates: register pressure, vectorization, ISA.

Split from pressure.py.
"""

from __future__ import annotations

from generator_ir.templates.stress_sync import (
    _SharedTemplate, _PRESSURE_LEVELS, _kx_mix_constant, _u, _mod, _tid,
)
from typing import Any, Dict, List, Set

from generator_ir.ir.typedecl import (
    T, TArray,
)
from generator_ir.ir.operators import Arith, Cmp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, BinOp, RelOp, Cast, SafeBin, Subscript,
    Call, Fma,
)
from generator_ir.ir.statements import (
    Stmt, Assign, Declare, Block, If, For, Barrier, Fence, RawCode,
)
from generator_ir.ir.arch import Arch
from generator_ir.gen.context import GenContext


class RegisterPressureTemplate(_SharedTemplate):
    """N live scalars with cross-mixing rounds (spill/fill stress)."""

    def __init__(self) -> None:
        self._pressure_level = "medium"
        self._block_size = 256

    @property
    def name(self) -> str:
        return "reg_spill_probe"

    @property
    def features(self) -> Set[str]:
        return {"register_pressure", "sync", "shared_mem"}

    def fetch_params(self) -> Dict[str, Any]:
        return {
            "pressure_level": self._pressure_level,
            "block_size": self._block_size,
        }

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        if self.kx_flat_mode(ctx):
            return self.kx_flat_body(ctx)
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        level = ctx.rnd_choice(list(_PRESSURE_LEVELS.keys()))
        n_vars = _PRESSURE_LEVELS[level]
        use_mix2 = n_vars >= 48
        use_barrier = n_vars >= 80
        use_shm = n_vars >= 80

        stmts: List[Stmt] = []
        smem = None
        if use_shm:
            _, smem, decl = self._declare_shared(ctx, "rp_smem", bs)
            stmts.append(decl)

        rp: List[Var] = []
        for i in range(n_vars):
            vn = ctx.kx_fresh_name(f"rp{i}")
            v = Var(T.u32, name=vn)
            ctx.kx_declare(v)
            idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid, rhs=Lit(T.u32, value=i)), n_u)
            stmts.append(Declare(name=vn, gtype=T.u32,
                                 init=BinOp(T.u32, op=Arith.BXOR,
                                            lhs=Subscript(T.u32, array=input_var, index=idx),
                                            rhs=Lit(T.u32, value=_kx_mix_constant(i)))))
            rp.append(v)

        def cross_mix(a: int, b: int, op: Arith) -> None:
            for i in range(n_vars):
                j1 = (i + a) % n_vars
                j2 = (i + b) % n_vars
                stmts.append(Assign(rp[i], BinOp(T.u32, op=Arith.BXOR,
                    lhs=rp[i], rhs=BinOp(T.u32, op=op, lhs=rp[j1], rhs=rp[j2]))))

        cross_mix(1, 7, Arith.ADD)

        if use_barrier and smem is not None:
            stmts.append(Assign(
                Subscript(T.u32, array=smem, index=_mod(tid, Lit(T.u32, value=bs))),
                BinOp(T.u32, op=Arith.BXOR, lhs=rp[0], rhs=rp[n_vars - 1])))
            stmts.append(Fence(kx_scope="block"))
            stmts.append(Barrier())

        if use_mix2:
            cross_mix(3, 11, Arith.MUL)

        # Post-barrier consume: read back the anchor a neighbour staged and
        # mix it into the last register so the barrier write has a consumer.
        if use_barrier and smem is not None:
            neighbor = _mod(BinOp(T.u32, op=Arith.ADD, lhs=_tid(),
                                  rhs=Lit(T.u32, value=1)),
                            Lit(T.u32, value=bs))
            stmts.append(Assign(rp[n_vars - 1],
                BinOp(T.u32, op=Arith.BXOR,
                      lhs=rp[n_vars - 1],
                      rhs=Subscript(T.u32, array=smem, index=neighbor))))

        # Fold all registers into one accumulator (none can be DCE'd).
        acc_name = ctx.kx_fresh_name("rp_acc")
        acc = Var(T.u32, name=acc_name)
        ctx.kx_declare(acc)
        result: Expr = rp[0]
        for v in rp[1:]:
            result = BinOp(T.u32, op=Arith.BXOR, lhs=result, rhs=v)
        stmts.append(Declare(name=acc_name, gtype=T.u32, init=result))

        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid), acc)])))
        return stmts

    def kx_flat_body(self, ctx: GenContext) -> List[Stmt]:
        """observable_fold: keep the real register-pressure structure (N live
        scalars + cross-mixing + barrier-spanning shared anchor) but drop the
        `if(tid<n)` write wrapper and fold into a sampled pre-template pool
        variable; a trailing barrier keeps the anchor reusable."""
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        level = ctx.rnd_choice(list(_PRESSURE_LEVELS.keys()))
        n_vars = _PRESSURE_LEVELS[level]
        use_mix2 = n_vars >= 48
        use_barrier = n_vars >= 80
        use_shm = n_vars >= 80

        # Sample the result sink only from variables that existed before this
        # template (never its own rp* scratch).
        pool_snapshot = ctx.kx_visible_declared_names()

        stmts: List[Stmt] = []
        smem = None
        if use_shm:
            _, smem, decl = self._declare_shared(ctx, "rp_smem", bs)
            stmts.append(decl)

        rp: List[Var] = []
        for i in range(n_vars):
            vn = ctx.kx_fresh_name(f"rp{i}")
            v = Var(T.u32, name=vn)
            ctx.kx_declare(v)
            idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid, rhs=Lit(T.u32, value=i)), n_u)
            stmts.append(Declare(name=vn, gtype=T.u32,
                                 init=BinOp(T.u32, op=Arith.BXOR,
                                            lhs=Subscript(T.u32, array=input_var, index=idx),
                                            rhs=Lit(T.u32, value=_kx_mix_constant(i)))))
            rp.append(v)

        def cross_mix(a: int, b: int, op: Arith) -> None:
            for i in range(n_vars):
                j1 = (i + a) % n_vars
                j2 = (i + b) % n_vars
                stmts.append(Assign(rp[i], BinOp(T.u32, op=Arith.BXOR,
                    lhs=rp[i], rhs=BinOp(T.u32, op=op, lhs=rp[j1], rhs=rp[j2]))))

        cross_mix(1, 7, Arith.ADD)

        if use_barrier and smem is not None:
            # Unconditional shared write keeps the barrier uniform.
            stmts.append(Assign(
                Subscript(T.u32, array=smem, index=_mod(tid, Lit(T.u32, value=bs))),
                BinOp(T.u32, op=Arith.BXOR, lhs=rp[0], rhs=rp[n_vars - 1])))
            stmts.append(Fence(kx_scope="block"))
            stmts.append(Barrier())

        if use_mix2:
            cross_mix(3, 11, Arith.MUL)

        if use_barrier and smem is not None:
            neighbor = _mod(BinOp(T.u32, op=Arith.ADD, lhs=_tid(),
                                  rhs=Lit(T.u32, value=1)),
                            Lit(T.u32, value=bs))
            stmts.append(Assign(rp[n_vars - 1],
                BinOp(T.u32, op=Arith.BXOR,
                      lhs=rp[n_vars - 1],
                      rhs=Subscript(T.u32, array=smem, index=neighbor))))

        result: Expr = rp[0]
        for v in rp[1:]:
            result = BinOp(T.u32, op=Arith.BXOR, lhs=result, rhs=v)

        saved_visible = ctx._visible_names
        ctx._visible_names = pool_snapshot
        try:
            sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
        finally:
            ctx._visible_names = saved_visible
        if sink_decl is not None:
            stmts.append(sink_decl)
        stmts.append(Assign(sink, result))
        # Reusable anchor on the next repeat-loop iteration.
        if use_barrier and smem is not None:
            stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# Vectorization stress
# ---------------------------------------------------------------------------

# Odd Goldbach/Weyl mixing constants: multiplies stay invertible mod 2^32 so
# the compiler cannot simplify the chains away.
_GOLDEN = 0x9E3779B9
_MIX_A = 0x6B9B0B55
_MIX_B = 0x517CC1B7

_EXTRA_INIT_CONSTANTS = [
    0xDEADBEEF, 0xCAFEBABE, 0xFEEDFACE, 0xABADCAFE,
    0xC0DEBABE, 0xB00BFACE, 0xDECAFBAD, 0xFACEB00C,
    0x0BADC0DE, 0xBADF00D0, 0x5A5A5A5B, 0xA5A5A5A6,
    0x12345679, 0xFEDCBA99, 0x87654322, 0x76543211,
]


class VectorizationStressTemplate(_SharedTemplate):
    """Consecutive 4-wide loads (LDG.128-eligible) with mixed offsets."""

    def __init__(self) -> None:
        self._difficulty = "medium"
        self._block_size = 256

    @property
    def name(self) -> str:
        return "vec_load_probe"

    @property
    def features(self) -> Set[str]:
        return {"coalesced_access", "vectorization", "sync"}

    def fetch_params(self) -> Dict[str, Any]:
        return {
            "difficulty": self._difficulty,
            "block_size": self._block_size,
        }

    def kx_required_shared_mem(self) -> int:
        # 4 uint slots per thread for the barrier-spanning section.
        return self._block_size * 4 * 4

    def build_body(self, ctx: GenContext) -> List[Stmt]:  # noqa: C901
        if self.kx_flat_mode(ctx):
            return self.kx_flat_body(ctx)
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        difficulty = ctx.rnd_choice(["easy", "medium", "hard"])

        stmts: List[Stmt] = []
        _, smem, decl = self._declare_shared(ctx, "vs_smem", bs * 4)
        stmts.append(decl)

        # base = tid * 4;  the 4-wide load stays in bounds when
        # base + 3 < n  ⟺  tid < (n - 3) / 4  (safe division for n < 4).
        base_name = ctx.kx_fresh_name("vs_base")
        base = Var(T.u32, name=base_name)
        ctx.kx_declare(base)
        stmts.append(Declare(name=base_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.MUL, lhs=tid,
                                        rhs=Lit(T.u32, value=4))))
        thresh_name = ctx.kx_fresh_name("vs_thresh")
        thresh = Var(T.u32, name=thresh_name)
        ctx.kx_declare(thresh)
        n_minus3 = BinOp(T.i32, op=Arith.SUB, lhs=n_var, rhs=Lit(T.i32, value=3))
        stmts.append(Declare(name=thresh_name, gtype=T.u32,
                             init=SafeBin(T.u32, op=SafeOp.SDIV,
                                          lhs=_u(n_minus3),
                                          rhs=Lit(T.u32, value=4))))

        def _fetch(off: int) -> Subscript:
            idx: Expr = base if off == 0 else BinOp(
                T.u32, op=Arith.ADD, lhs=base, rhs=Lit(T.u32, value=off))
            return Subscript(T.u32, array=input_var, index=idx)

        result_name = ctx.kx_fresh_name("vs_result")
        result_var = Var(T.u32, name=result_name)
        ctx.kx_declare(result_var)
        stmts.append(Declare(name=result_name, gtype=T.u32,
                             init=Lit(T.u32, value=0)))

        # Load/compute vars are declared at OUTER kx_scope (init 0, assigned
        # inside the guard) so they stay visible to the checksum fold.
        guarded: List[Stmt] = []
        if difficulty == "easy":
            # 4 back-to-back loads; a conformant optimiser merges to LDG.128.
            v_vars: List[Var] = []
            for k in range(4):
                vn = ctx.kx_fresh_name(f"v{k}")
                vv = Var(T.u32, name=vn)
                ctx.kx_declare(vv)
                stmts.append(Declare(name=vn, gtype=T.u32,
                                     init=Lit(T.u32, value=0)))
                v_vars.append(vv)
            for k, vv in enumerate(v_vars):
                guarded.append(Assign(vv, _fetch(k)))
            fold: Expr = v_vars[0]
            for vv in v_vars[1:]:
                fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=vv)
            guarded.append(Assign(result_var, fold))
        elif difficulty == "medium":
            # load/compute interleave: the chain between loads tests whether
            # alias analysis can still merge them.
            tmp_name = ctx.kx_fresh_name("vs_tmp")
            tmp = Var(T.u32, name=tmp_name)
            ctx.kx_declare(tmp)
            stmts.append(Declare(name=tmp_name, gtype=T.u32,
                                 init=Lit(T.u32, value=0)))
            v_vars = []
            for k in range(4):
                vn = ctx.kx_fresh_name(f"v{k}")
                vv = Var(T.u32, name=vn)
                ctx.kx_declare(vv)
                stmts.append(Declare(name=vn, gtype=T.u32,
                                     init=Lit(T.u32, value=0)))
                v_vars.append(vv)
            guarded.append(Assign(v_vars[0], _fetch(0)))
            guarded.append(Assign(tmp, BinOp(T.u32, op=Arith.MUL,
                                             lhs=v_vars[0],
                                             rhs=Lit(T.u32, value=_GOLDEN))))
            for k in range(1, 4):
                guarded.append(Assign(v_vars[k], _fetch(k)))
                guarded.append(Assign(tmp, BinOp(T.u32, op=Arith.BXOR,
                    lhs=tmp,
                    rhs=BinOp(T.u32, op=Arith.MUL, lhs=v_vars[k],
                              rhs=Lit(T.u32, value=_GOLDEN ^ (k * 0x1111))))))
            guarded.append(Assign(result_var, tmp))
        else:
            # hard: 4 consecutive loads + 16 competing live scalars.
            v_vars = []
            for k in range(4):
                vn = ctx.kx_fresh_name(f"hv{k}")
                vv = Var(T.u32, name=vn)
                ctx.kx_declare(vv)
                stmts.append(Declare(name=vn, gtype=T.u32,
                                     init=Lit(T.u32, value=0)))
                v_vars.append(vv)
            for k, vv in enumerate(v_vars):
                guarded.append(Assign(vv, _fetch(k)))
            n_extra = 16
            extra: List[Var] = []
            for e in range(n_extra):
                en = ctx.kx_fresh_name(f"he{e}")
                ev = Var(T.u32, name=en)
                ctx.kx_declare(ev)
                stmts.append(Declare(name=en, gtype=T.u32,
                                     init=Lit(T.u32, value=0)))
                extra.append(ev)
            for e in range(n_extra):
                src = v_vars[e % 4]
                guarded.append(Assign(extra[e],
                    BinOp(T.u32, op=Arith.MUL,
                          lhs=BinOp(T.u32, op=Arith.BXOR, lhs=src,
                                    rhs=Lit(T.u32,
                                            value=_EXTRA_INIT_CONSTANTS[e])),
                          rhs=Lit(T.u32, value=_GOLDEN ^ (e * 0x2222 + 1)))))
            for i in range(n_extra):
                j1 = (i + 1) % n_extra
                j5 = (i + 5) % n_extra
                guarded.append(Assign(extra[i],
                    BinOp(T.u32, op=Arith.BXOR, lhs=extra[i],
                          rhs=BinOp(T.u32, op=Arith.BXOR, lhs=extra[j1],
                                    rhs=BinOp(T.u32, op=Arith.MUL,
                                              lhs=extra[j5],
                                              rhs=Lit(T.u32, value=_MIX_B))))))
            fold = v_vars[0]
            for vv in v_vars[1:]:
                fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=vv)
            for ev in extra:
                fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=ev)
            guarded.append(Assign(result_var, fold))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=thresh),
                        then=Block(guarded)))

        # Barrier-spanning section (always emitted): four loads before the
        # barrier are consumed after it, together with the values a neighbour
        # staged in shared memory.
        lane = _tid()
        smem_base = BinOp(T.u32, op=Arith.MUL, lhs=lane, rhs=Lit(T.u32, value=4))
        pre: List[Var] = []
        for k in range(4):
            pn = ctx.kx_fresh_name(f"pre{k}")
            pv = Var(T.u32, name=pn)
            ctx.kx_declare(pv)
            safe_idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid,
                                  rhs=Lit(T.u32, value=bs * k + k + 1)), n_u)
            stmts.append(Declare(name=pn, gtype=T.u32,
                                 init=Subscript(T.u32, array=input_var,
                                                index=safe_idx)))
            pre.append(pv)
            stmts.append(Assign(
                Subscript(T.u32, array=smem,
                          index=BinOp(T.u32, op=Arith.ADD, lhs=smem_base,
                                      rhs=Lit(T.u32, value=k))),
                BinOp(T.u32, op=Arith.BXOR, lhs=pv,
                      rhs=Lit(T.u32, value=_MIX_A ^ k))))
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        nb_name = ctx.kx_fresh_name("vs_nb_base")
        nb_base = Var(T.u32, name=nb_name)
        ctx.kx_declare(nb_base)
        neighbor = _mod(BinOp(T.u32, op=Arith.ADD, lhs=lane,
                              rhs=Lit(T.u32, value=1)),
                        Lit(T.u32, value=bs))
        stmts.append(Declare(name=nb_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.MUL, lhs=neighbor,
                                        rhs=Lit(T.u32, value=4))))
        post: Expr = result_var
        for k in range(4):
            nb_val = Subscript(T.u32, array=smem,
                               index=BinOp(T.u32, op=Arith.ADD, lhs=nb_base,
                                           rhs=Lit(T.u32, value=k)))
            post = BinOp(T.u32, op=Arith.BXOR, lhs=post,
                         rhs=BinOp(T.u32, op=Arith.BXOR, lhs=nb_val,
                                   rhs=pre[k]))
        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            post)])))
        # Keep vs_smem reusable on the next repeat-loop iteration.
        stmts.append(Barrier())
        return stmts

    def kx_flat_body(self, ctx: GenContext) -> List[Stmt]:  # noqa: C901
        """observable_fold: keep the real load-merging structure (4 consecutive
        loads + difficulty compute + barrier-spanning section) but bound every
        load via SAFE_MOD, drop the `tid < thresh` guard, and fold into a
        sampled pre-template pool variable."""
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)
        bs = self._block_size
        difficulty = ctx.rnd_choice(["easy", "medium", "hard"])

        # Sample the result sink only from variables that existed before this
        # template (never its own vs_*/v*/hv*/he*/pre* scratch).
        pool_snapshot = ctx.kx_visible_declared_names()

        stmts: List[Stmt] = []
        _, smem, decl = self._declare_shared(ctx, "vs_smem", bs * 4)
        stmts.append(decl)

        base_name = ctx.kx_fresh_name("vs_base")
        base = Var(T.u32, name=base_name)
        ctx.kx_declare(base)
        stmts.append(Declare(name=base_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.MUL, lhs=tid,
                                        rhs=Lit(T.u32, value=4))))

        def _fetch(off: int) -> Subscript:
            idx: Expr = base if off == 0 else BinOp(
                T.u32, op=Arith.ADD, lhs=base, rhs=Lit(T.u32, value=off))
            return Subscript(T.u32, array=input_var, index=_mod(idx, n_u))

        result_name = ctx.kx_fresh_name("vs_result")
        result_var = Var(T.u32, name=result_name)
        ctx.kx_declare(result_var)

        if difficulty == "easy":
            v_vars: List[Var] = []
            for k in range(4):
                vn = ctx.kx_fresh_name(f"v{k}")
                vv = Var(T.u32, name=vn)
                ctx.kx_declare(vv)
                stmts.append(Declare(name=vn, gtype=T.u32, init=_fetch(k)))
                v_vars.append(vv)
            fold: Expr = v_vars[0]
            for vv in v_vars[1:]:
                fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=vv)
            stmts.append(Declare(name=result_name, gtype=T.u32, init=fold))
        elif difficulty == "medium":
            v_vars = []
            for k in range(4):
                vn = ctx.kx_fresh_name(f"v{k}")
                vv = Var(T.u32, name=vn)
                ctx.kx_declare(vv)
                stmts.append(Declare(name=vn, gtype=T.u32, init=_fetch(k)))
                v_vars.append(vv)
            tmp_name = ctx.kx_fresh_name("vs_tmp")
            tmp = Var(T.u32, name=tmp_name)
            ctx.kx_declare(tmp)
            stmts.append(Declare(name=tmp_name, gtype=T.u32,
                                 init=BinOp(T.u32, op=Arith.MUL,
                                            lhs=v_vars[0],
                                            rhs=Lit(T.u32, value=_GOLDEN))))
            for k in range(1, 4):
                stmts.append(Assign(tmp, BinOp(T.u32, op=Arith.BXOR,
                    lhs=tmp,
                    rhs=BinOp(T.u32, op=Arith.MUL, lhs=v_vars[k],
                              rhs=Lit(T.u32, value=_GOLDEN ^ (k * 0x1111))))))
            stmts.append(Declare(name=result_name, gtype=T.u32, init=tmp))
        else:
            v_vars = []
            for k in range(4):
                vn = ctx.kx_fresh_name(f"hv{k}")
                vv = Var(T.u32, name=vn)
                ctx.kx_declare(vv)
                stmts.append(Declare(name=vn, gtype=T.u32, init=_fetch(k)))
                v_vars.append(vv)
            n_extra = 16
            extra: List[Var] = []
            for e in range(n_extra):
                en = ctx.kx_fresh_name(f"he{e}")
                ev = Var(T.u32, name=en)
                ctx.kx_declare(ev)
                src = v_vars[e % 4]
                stmts.append(Declare(name=en, gtype=T.u32,
                    init=BinOp(T.u32, op=Arith.MUL,
                               lhs=BinOp(T.u32, op=Arith.BXOR, lhs=src,
                                         rhs=Lit(T.u32,
                                                 value=_EXTRA_INIT_CONSTANTS[e])),
                               rhs=Lit(T.u32,
                                       value=_GOLDEN ^ (e * 0x2222 + 1)))))
                extra.append(ev)
            for i in range(n_extra):
                j1 = (i + 1) % n_extra
                j5 = (i + 5) % n_extra
                stmts.append(Assign(extra[i],
                    BinOp(T.u32, op=Arith.BXOR, lhs=extra[i],
                          rhs=BinOp(T.u32, op=Arith.BXOR, lhs=extra[j1],
                                    rhs=BinOp(T.u32, op=Arith.MUL,
                                              lhs=extra[j5],
                                              rhs=Lit(T.u32, value=_MIX_B))))))
            fold = v_vars[0]
            for vv in v_vars[1:]:
                fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=vv)
            for ev in extra:
                fold = BinOp(T.u32, op=Arith.BXOR, lhs=fold, rhs=ev)
            stmts.append(Declare(name=result_name, gtype=T.u32, init=fold))

        # Barrier-spanning section (always emitted; the shared write
        # smem[lane*4 + k] is unconditional, so the barrier is uniform).
        lane = _tid()
        smem_base = BinOp(T.u32, op=Arith.MUL, lhs=lane, rhs=Lit(T.u32, value=4))
        pre: List[Var] = []
        for k in range(4):
            pn = ctx.kx_fresh_name(f"pre{k}")
            pv = Var(T.u32, name=pn)
            ctx.kx_declare(pv)
            safe_idx = _mod(BinOp(T.u32, op=Arith.ADD, lhs=tid,
                                  rhs=Lit(T.u32, value=bs * k + k + 1)), n_u)
            stmts.append(Declare(name=pn, gtype=T.u32,
                                 init=Subscript(T.u32, array=input_var,
                                                index=safe_idx)))
            pre.append(pv)
            stmts.append(Assign(
                Subscript(T.u32, array=smem,
                          index=BinOp(T.u32, op=Arith.ADD, lhs=smem_base,
                                      rhs=Lit(T.u32, value=k))),
                BinOp(T.u32, op=Arith.BXOR, lhs=pv,
                      rhs=Lit(T.u32, value=_MIX_A ^ k))))
        stmts.append(Fence(kx_scope="block"))
        stmts.append(Barrier())

        nb_name = ctx.kx_fresh_name("vs_nb_base")
        nb_base = Var(T.u32, name=nb_name)
        ctx.kx_declare(nb_base)
        neighbor = _mod(BinOp(T.u32, op=Arith.ADD, lhs=lane,
                              rhs=Lit(T.u32, value=1)),
                        Lit(T.u32, value=bs))
        stmts.append(Declare(name=nb_name, gtype=T.u32,
                             init=BinOp(T.u32, op=Arith.MUL, lhs=neighbor,
                                        rhs=Lit(T.u32, value=4))))
        post: Expr = result_var
        for k in range(4):
            nb_val = Subscript(T.u32, array=smem,
                               index=BinOp(T.u32, op=Arith.ADD, lhs=nb_base,
                                           rhs=Lit(T.u32, value=k)))
            post = BinOp(T.u32, op=Arith.BXOR, lhs=post,
                         rhs=BinOp(T.u32, op=Arith.BXOR, lhs=nb_val,
                                   rhs=pre[k]))

        saved_visible = ctx._visible_names
        ctx._visible_names = pool_snapshot
        try:
            sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
        finally:
            ctx._visible_names = saved_visible
        if sink_decl is not None:
            stmts.append(sink_decl)
        stmts.append(Assign(sink, post))
        # Make vs_smem safe to reuse on the next repeat-loop iteration.
        stmts.append(Barrier())
        return stmts


# ---------------------------------------------------------------------------
# PTX stress (instruction-selection paths)
# ---------------------------------------------------------------------------


def _i(v: int) -> Lit:
    return Lit(T.i32, value=v)


def _i32(v: Expr) -> Cast:
    return Cast(T.i32, to=T.i32, arg=v)


def _f32(v: Expr) -> Cast:
    return Cast(T.f32, to=T.f32, arg=v)


def _kx_xor(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.BXOR, lhs=a, rhs=b)


def _kx_add(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.ADD, lhs=a, rhs=b)


def _mul(a: Expr, b: Expr) -> BinOp:
    return BinOp(T.u32, op=Arith.MUL, lhs=a, rhs=b)


def _f_as_i(v: Expr) -> Cast:
    """Float bit-pattern to u32 via __float_as_int."""
    return _u(Call(T.i32, fn="__float_as_int", args=[v], result_ty=T.i32))


class PTXStressTemplate(_SharedTemplate):
    """Instruction-selection stress: a seeded sample of 3-6 pattern groups
    (wide-mul / bit-ops / rounding / float edges / int intrinsics / half /
    local array / register spill) folded into one per-thread accumulator."""

    # All available pattern groups -- ordered for stable random selection.
    _ALL_PATTERNS: List[str] = [
        "wide_mul",
        "bit_ops",
        "rounding_modes",
        "float_edges",
        "int_intrinsics",
        "half_ops",
        "local_array",
        "reg_spill",
    ]

    # Integer-only groups, bit-stable across nvcc versions (no float/half
    # intrinsics): used by the deterministic-checksum flat path.
    _FLAT_PATTERNS: List[str] = [
        "wide_mul",
        "bit_ops",
        "int_intrinsics",
        "local_array",
        "reg_spill",
    ]

    def __init__(self) -> None:
        self._block_size = 256
        # Number of pattern groups to select per generation (3-6).
        self._num_patterns = 4

    @property
    def name(self) -> str:
        return "isa_stress"

    @property
    def features(self) -> Set[str]:
        return {
            "int_math",
            "bit_manip",
            "instruction_stress",
            "wide_multiply",
            "rounding_modes",
            "float_math",
            "half_precision",
            "register_pressure",
        }

    def fetch_params(self) -> Dict[str, Any]:
        return {
            "block_size": self._block_size,
            "num_patterns": self._num_patterns,
        }

    def kx_required_shared_mem(self) -> int:
        # All computation is thread-local; no shared memory required.
        return 0

    def build_body(self, ctx: GenContext) -> List[Stmt]:
        if self.kx_flat_mode(ctx):
            return self.kx_flat_body(ctx)
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)

        stmts: List[Stmt] = []
        acc = self._seed_acc(ctx, stmts, tid, n_u, input_var)

        arch, kx_is_cuda, sm = self._target(ctx)
        available = list(self._ALL_PATTERNS)
        # Half-precision patterns require sm_53+ on NVIDIA (CUDA only).
        if not (kx_is_cuda and arch.kx_is_nvidia and sm >= 53):
            available = [p for p in available if p != "half_ops"]
        k = min(self._num_patterns, len(available))
        k = max(k, min(3, len(available)))
        selected = ctx.rng.sample(available, k)
        for pattern in selected:
            stmts.extend(self._kx_dispatch(pattern, ctx, acc, tid, n_u,
                                        input_var, kx_is_cuda, sm))

        stmts.append(If(cond=RelOp(T.boolean, op=Cmp.LT, lhs=tid, rhs=n_u),
                        then=Block([Assign(
                            Subscript(T.u32, array=output_var, index=tid),
                            acc)])))
        return stmts

    def kx_flat_body(self, ctx: GenContext) -> List[Stmt]:
        """observable_fold: restrict selection to the bit-stable INTEGER
        groups (float/half/rounding-mode bit patterns are not guaranteed
        identical across nvcc versions) and fold the accumulator into a
        sampled pre-template pool variable.  No shared memory, no barriers."""
        tid, n_var, output_var, input_var, n_u = self._scope(ctx)

        # Sample the result sink only from variables that existed before this
        # template (never its own ptx_acc/wm_*/bo_*/... scratch).
        pool_snapshot = ctx.kx_visible_declared_names()

        stmts: List[Stmt] = []
        acc = self._seed_acc(ctx, stmts, tid, n_u, input_var)

        arch, kx_is_cuda, sm = self._target(ctx)
        available = list(self._FLAT_PATTERNS)
        k = min(self._num_patterns, len(available))
        k = max(k, min(3, len(available)))
        selected = ctx.rng.sample(available, k)
        for pattern in selected:
            stmts.extend(self._kx_dispatch(pattern, ctx, acc, tid, n_u,
                                        input_var, kx_is_cuda, sm))

        saved_visible = ctx._visible_names
        ctx._visible_names = pool_snapshot
        try:
            sink, sink_decl = ctx.kx_sample_or_create(T.u32, for_write=True)
        finally:
            ctx._visible_names = saved_visible
        if sink_decl is not None:
            stmts.append(sink_decl)
        stmts.append(Assign(sink, _kx_xor(sink, acc)))
        return stmts

    # ------------------------------------------------------------------
    # shared plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _target(ctx: GenContext):
        arch = Arch.parse(getattr(ctx.config, "isa", "sm_75"))
        kx_is_cuda = getattr(ctx.config, "target", "hip") == "cuda"
        return arch, kx_is_cuda, arch.version

    def _seed_acc(self, ctx: GenContext, stmts: List[Stmt], tid: Var,
                  n_u: Expr, input_var: Var) -> Var:
        """Main accumulator, seeded from two bounded input slots."""
        acc_name = ctx.kx_fresh_name("ptx_acc")
        acc = Var(T.u32, name=acc_name)
        ctx.kx_declare(acc)
        seed = _mod(tid, n_u)
        seed2 = _mod(_kx_add(tid, Lit(T.u32, value=7)), n_u)
        init = _kx_xor(Subscript(T.u32, array=input_var, index=seed),
                    _mul(Subscript(T.u32, array=input_var, index=seed2),
                         Lit(T.u32, value=0x9E3779B9)))
        stmts.append(Declare(name=acc_name, gtype=T.u32, init=init))
        return acc

    def _kx_dispatch(self, pattern: str, ctx: GenContext, acc: Var, tid: Var,
                  n_u: Expr, input_var: Var, kx_is_cuda: bool,
                  sm: int) -> List[Stmt]:
        dispatch = {
            "wide_mul": self._build_wide_mul,
            "bit_ops": self._build_bit_ops,
            "rounding_modes": self._build_rounding_modes,
            "float_edges": self._build_float_edges,
            "int_intrinsics": self._build_int_intrinsics,
            "half_ops": self._build_half_ops,
            "local_array": self._build_local_array,
            "reg_spill": self._build_register_spill,
        }
        fn = dispatch.get(pattern)
        if fn is None:
            return []
        return fn(ctx, acc, tid, n_u, input_var, kx_is_cuda, sm)

    # ------------------------------------------------------------------
    # 1. Wide multiply
    # ------------------------------------------------------------------

    def _build_wide_mul(self, ctx: GenContext, acc: Var, tid: Var, n_u: Expr,
                      input_var: Var, kx_is_cuda: bool, sm: int) -> List[Stmt]:
        """__umulhi plus the signed/widening 64-bit multiply-high paths
        (IMAD.HI / MULHI and the widening 32x32->64 selection)."""
        stmts: List[Stmt] = []
        a_name = ctx.kx_fresh_name("wm_a")
        b_name = ctx.kx_fresh_name("wm_b")
        a = Var(T.u32, name=a_name)
        b = Var(T.u32, name=b_name)
        ctx.kx_declare(a)
        ctx.kx_declare(b)
        idx_a = _mod(_kx_add(tid, Lit(T.u32, value=3)), n_u)
        idx_b = _mod(_kx_add(tid, Lit(T.u32, value=11)), n_u)
        stmts.append(Declare(name=a_name, gtype=T.u32,
                             init=Subscript(T.u32, array=input_var,
                                            index=idx_a)))
        stmts.append(Declare(name=b_name, gtype=T.u32,
                             init=Subscript(T.u32, array=input_var,
                                            index=idx_b)))

        hi_u_name = ctx.kx_fresh_name("wm_hi_u")
        hi_u = Var(T.u32, name=hi_u_name)
        ctx.kx_declare(hi_u)
        stmts.append(Declare(name=hi_u_name, gtype=T.u32,
                             init=Call(T.u32, fn="__umulhi", args=[a, b],
                                       result_ty=T.u32)))

        # Signed multiply-high via the compiler-invariant 64-bit path:
        # nvcc's signed __mulhi diverges across versions for negative inputs,
        # so widen through i64/u64 and take the logical high word instead.
        hi_s_name = ctx.kx_fresh_name("wm_hi_s")
        hi_s = Var(T.u32, name=hi_s_name)
        ctx.kx_declare(hi_s)
        prod_s = BinOp(T.u64, op=Arith.MUL,
                       lhs=Cast(T.u64, to=T.u64,
                                arg=Cast(T.i64, to=T.i64, arg=_i32(a))),
                       rhs=Cast(T.u64, to=T.u64,
                                arg=Cast(T.i64, to=T.i64, arg=_i32(b))))
        stmts.append(Declare(name=hi_s_name, gtype=T.u32,
                             init=_u(SafeBin(T.u64, op=SafeOp.SSHR, lhs=prod_s,
                                             rhs=Cast(T.u64, to=T.u64,
                                                      arg=Lit(T.u32,
                                                              value=32))))))

        # Widening multiply: cast to u64, multiply, extract the upper word.
        wide_name = ctx.kx_fresh_name("wm_wide")
        wide = Var(T.u64, name=wide_name)
        ctx.kx_declare(wide)
        stmts.append(Declare(name=wide_name, gtype=T.u64,
                             init=BinOp(T.u64, op=Arith.MUL,
                                        lhs=Cast(T.u64, to=T.u64, arg=a),
                                        rhs=Cast(T.u64, to=T.u64, arg=b))))
        wide_hi_name = ctx.kx_fresh_name("wm_wide_hi")
        wide_hi = Var(T.u32, name=wide_hi_name)
        ctx.kx_declare(wide_hi)
        stmts.append(Declare(name=wide_hi_name, gtype=T.u32,
                             init=_u(SafeBin(T.u64, op=SafeOp.SSHR, lhs=wide,
                                             rhs=Cast(T.u64, to=T.u64,
                                                      arg=Lit(T.u32,
                                                              value=32))))))

        stmts.append(Assign(acc, _kx_xor(acc, _kx_xor(_kx_xor(hi_u, hi_s), wide_hi))))
        return stmts

    # ------------------------------------------------------------------
    # 2. Bit manipulation
    # ------------------------------------------------------------------

    def _build_bit_ops(self, ctx: GenContext, acc: Var, tid: Var, n_u: Expr,
                     input_var: Var, kx_is_cuda: bool, sm: int) -> List[Stmt]:
        """__brev/__popc/__clz/__ffs/__byte_perm chains (BREV/POPC/CLZ/FFS and
        the BYTE_PERM register-layout shuffle)."""
        stmts: List[Stmt] = []
        x_name = ctx.kx_fresh_name("bo_x")
        y_name = ctx.kx_fresh_name("bo_y")
        x = Var(T.u32, name=x_name)
        y = Var(T.u32, name=y_name)
        ctx.kx_declare(x)
        ctx.kx_declare(y)
        idx_x = _mod(_kx_add(tid, Lit(T.u32, value=5)), n_u)
        idx_y = _mod(_kx_add(tid, Lit(T.u32, value=19)), n_u)
        stmts.append(Declare(name=x_name, gtype=T.u32,
                             init=Subscript(T.u32, array=input_var,
                                            index=idx_x)))
        stmts.append(Declare(name=y_name, gtype=T.u32,
                             init=Subscript(T.u32, array=input_var,
                                            index=idx_y)))

        brev_name = ctx.kx_fresh_name("bo_brev")
        brev = Var(T.u32, name=brev_name)
        ctx.kx_declare(brev)
        inner = _kx_xor(Call(T.u32, fn="__popc", args=[x], result_ty=T.u32),
                     Call(T.u32, fn="__clz", args=[y], result_ty=T.u32))
        stmts.append(Declare(name=brev_name, gtype=T.u32,
                             init=Call(T.u32, fn="__brev", args=[inner],
                                       result_ty=T.u32)))

        ffs_name = ctx.kx_fresh_name("bo_ffs")
        ffs = Var(T.u32, name=ffs_name)
        ctx.kx_declare(ffs)
        stmts.append(Declare(name=ffs_name, gtype=T.u32,
                             init=_u(Call(T.i32, fn="__ffs", args=[_kx_xor(x, y)],
                                          result_ty=T.i32))))

        sel_val = ctx.rnd_choice([0x0123, 0x3210, 0x1032, 0x2301, 0x0321])
        bp_name = ctx.kx_fresh_name("bo_bp")
        bp = Var(T.u32, name=bp_name)
        ctx.kx_declare(bp)
        stmts.append(Declare(name=bp_name, gtype=T.u32,
                             init=Call(T.u32, fn="__byte_perm",
                                       args=[x, y, Lit(T.u32, value=sel_val)],
                                       result_ty=T.u32)))

        stmts.append(Assign(acc, _kx_xor(acc, _kx_xor(brev, _kx_xor(ffs, bp)))))
        return stmts

    # ------------------------------------------------------------------
    # 3. Rounding-mode arithmetic
    # ------------------------------------------------------------------

    def _build_rounding_modes(self, ctx: GenContext, acc: Var, tid: Var,
                            n_u: Expr, input_var: Var, kx_is_cuda: bool,
                            sm: int) -> List[Stmt]:
        """__fadd_r{n,d,u,z} / __fmaf_rn on CUDA (FADD.R* selection); portable
        add/mul/fma fallback on HIP."""
        stmts: List[Stmt] = []
        fa_name = ctx.kx_fresh_name("rm_fa")
        fb_name = ctx.kx_fresh_name("rm_fb")
        fa = Var(T.f32, name=fa_name)
        fb = Var(T.f32, name=fb_name)
        ctx.kx_declare(fa)
        ctx.kx_declare(fb)
        idx_a = _mod(_kx_add(tid, Lit(T.u32, value=2)), n_u)
        idx_b = _mod(_kx_add(tid, Lit(T.u32, value=13)), n_u)
        raw_a = Subscript(T.u32, array=input_var, index=idx_a)
        raw_b = Subscript(T.u32, array=input_var, index=idx_b)

        if kx_is_cuda:
            # Reinterpret the uint bit patterns as floats.
            stmts.append(Declare(name=fa_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__int_as_float",
                                           args=[_i32(raw_a)],
                                           result_ty=T.f32)))
            stmts.append(Declare(name=fb_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__int_as_float",
                                           args=[_i32(raw_b)],
                                           result_ty=T.f32)))
            rn_name = ctx.kx_fresh_name("rm_rn")
            rd_name = ctx.kx_fresh_name("rm_rd")
            ru_name = ctx.kx_fresh_name("rm_ru")
            rz_name = ctx.kx_fresh_name("rm_rz")
            fmaf_name = ctx.kx_fresh_name("rm_fma")
            rn = Var(T.f32, name=rn_name)
            rd = Var(T.f32, name=rd_name)
            ru = Var(T.f32, name=ru_name)
            rz = Var(T.f32, name=rz_name)
            fmaf_v = Var(T.f32, name=fmaf_name)
            for v in (rn, rd, ru, rz, fmaf_v):
                ctx.kx_declare(v)
            stmts.append(Declare(name=rn_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__fadd_rn",
                                           args=[fa, fb], result_ty=T.f32)))
            stmts.append(Declare(name=rd_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__fadd_rd",
                                           args=[fa, fb], result_ty=T.f32)))
            stmts.append(Declare(name=ru_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__fadd_ru",
                                           args=[fa, fb], result_ty=T.f32)))
            stmts.append(Declare(name=rz_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__fadd_rz",
                                           args=[fa, fb], result_ty=T.f32)))
            stmts.append(Declare(name=fmaf_name, gtype=T.f32,
                                 init=Call(T.f32, fn="__fmaf_rn",
                                           args=[fa, fb, fa],
                                           result_ty=T.f32)))
            fold = _kx_xor(_f_as_i(rn),
                   _kx_xor(_f_as_i(rd),
                   _kx_xor(_f_as_i(ru),
                   _kx_xor(_f_as_i(rz), _f_as_i(fmaf_v)))))
            stmts.append(Assign(acc, _kx_xor(acc, fold)))
        else:
            # Portable path: mask to [0, 2^24) and use plain arithmetic.
            scale = Lit(T.u32, value=0x00FFFFFF)
            stmts.append(Declare(name=fa_name, gtype=T.f32,
                                 init=_f32(BinOp(T.u32, op=Arith.BAND,
                                                 lhs=raw_a, rhs=scale))))
            stmts.append(Declare(name=fb_name, gtype=T.f32,
                                 init=_f32(BinOp(T.u32, op=Arith.BAND,
                                                 lhs=raw_b, rhs=scale))))
            add_name = ctx.kx_fresh_name("rm_add")
            mul_name = ctx.kx_fresh_name("rm_mul")
            fma_name = ctx.kx_fresh_name("rm_fma")
            add_v = Var(T.f32, name=add_name)
            mul_v = Var(T.f32, name=mul_name)
            fma_v = Var(T.f32, name=fma_name)
            for v in (add_v, mul_v, fma_v):
                ctx.kx_declare(v)
            stmts.append(Declare(name=add_name, gtype=T.f32,
                                 init=BinOp(T.f32, op=Arith.ADD,
                                            lhs=fa, rhs=fb)))
            stmts.append(Declare(name=mul_name, gtype=T.f32,
                                 init=BinOp(T.f32, op=Arith.MUL,
                                            lhs=fa, rhs=fb)))
            stmts.append(Declare(name=fma_name, gtype=T.f32,
                                 init=Fma(T.f32, a=fa, b=fb, c=fa)))
            fold = _kx_xor(_f_as_i(add_v),
                   _kx_xor(_f_as_i(mul_v), _f_as_i(fma_v)))
            stmts.append(Assign(acc, _kx_xor(acc, fold)))
        return stmts

    # ------------------------------------------------------------------
    # 4. Float edge cases
    # ------------------------------------------------------------------

    def _build_float_edges(self, ctx: GenContext, acc: Var, tid: Var,
                         n_u: Expr, input_var: Var, kx_is_cuda: bool,
                         sm: int) -> List[Stmt]:
        """truncf/ceilf/floorf/roundf/fabsf/fmaxf/fminf/copysignf chains
        (FSEL / rounding built-in lowering)."""
        stmts: List[Stmt] = []
        fv_name = ctx.kx_fresh_name("fe_fv")
        fw_name = ctx.kx_fresh_name("fe_fw")
        fv = Var(T.f32, name=fv_name)
        fw = Var(T.f32, name=fw_name)
        ctx.kx_declare(fv)
        ctx.kx_declare(fw)
        idx_v = _mod(_kx_add(tid, Lit(T.u32, value=4)), n_u)
        idx_w = _mod(_kx_add(tid, Lit(T.u32, value=17)), n_u)
        # Scale to a reasonable float range: input & 0xFFFF gives [0..65535].
        raw_v = BinOp(T.u32, op=Arith.BAND,
                      lhs=Subscript(T.u32, array=input_var, index=idx_v),
                      rhs=Lit(T.u32, value=0xFFFF))
        raw_w = BinOp(T.u32, op=Arith.BAND,
                      lhs=Subscript(T.u32, array=input_var, index=idx_w),
                      rhs=Lit(T.u32, value=0xFFFF))
        stmts.append(Declare(name=fv_name, gtype=T.f32, init=_f32(raw_v)))
        stmts.append(Declare(name=fw_name, gtype=T.f32, init=_f32(raw_w)))

        names = {key: ctx.kx_fresh_name(f"fe_{key}") for key in
                 ("tr", "ce", "fl", "ro", "fa", "fmx", "fmn", "cp")}
        fvars: Dict[str, Var] = {}
        for key, vname in names.items():
            v = Var(T.f32, name=vname)
            ctx.kx_declare(v)
            fvars[key] = v
        stmts.append(Declare(name=names["tr"], gtype=T.f32,
                             init=Call(T.f32, fn="truncf", args=[fv],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["ce"], gtype=T.f32,
                             init=Call(T.f32, fn="ceilf", args=[fv],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["fl"], gtype=T.f32,
                             init=Call(T.f32, fn="floorf", args=[fv],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["ro"], gtype=T.f32,
                             init=Call(T.f32, fn="roundf", args=[fv],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["fa"], gtype=T.f32,
                             init=Call(T.f32, fn="fabsf",
                                       args=[BinOp(T.f32, op=Arith.SUB,
                                                   lhs=fv, rhs=fw)],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["fmx"], gtype=T.f32,
                             init=Call(T.f32, fn="fmaxf", args=[fv, fw],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["fmn"], gtype=T.f32,
                             init=Call(T.f32, fn="fminf", args=[fv, fw],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=names["cp"], gtype=T.f32,
                             init=Call(T.f32, fn="copysignf",
                                       args=[fvars["fmx"], fw],
                                       result_ty=T.f32)))

        fold: Expr = _f_as_i(fvars["tr"])
        for key in ("ce", "fl", "ro", "fa", "fmx", "fmn", "cp"):
            fold = _kx_xor(fold, _f_as_i(fvars[key]))
        stmts.append(Assign(acc, _kx_xor(acc, fold)))
        return stmts

    # ------------------------------------------------------------------
    # 5. Integer intrinsics
    # ------------------------------------------------------------------

    def _build_int_intrinsics(self, ctx: GenContext, acc: Var, tid: Var,
                            n_u: Expr, input_var: Var, kx_is_cuda: bool,
                            sm: int) -> List[Stmt]:
        """__usad, __mul24 and nested min/max chains (VABSDIFF / MUL24 /
        VMIN/VMAX selection)."""
        stmts: List[Stmt] = []
        names = [ctx.kx_fresh_name(p) for p in ("ii_a", "ii_b", "ii_c", "ii_d")]
        operands: List[Var] = []
        for vname, off in zip(names, (1, 5, 9, 23)):
            v = Var(T.u32, name=vname)
            ctx.kx_declare(v)
            operands.append(v)
            stmts.append(Declare(name=vname, gtype=T.u32,
                                 init=Subscript(T.u32, array=input_var,
                                                index=_mod(
                                                    _kx_add(tid, Lit(T.u32,
                                                                  value=off)),
                                                    n_u))))
        a, b, c, d = operands

        sad_name = ctx.kx_fresh_name("ii_sad")
        sad = Var(T.u32, name=sad_name)
        ctx.kx_declare(sad)
        stmts.append(Declare(name=sad_name, gtype=T.u32,
                             init=Call(T.u32, fn="__usad",
                                       args=[a, b, Lit(T.u32, value=0)],
                                       result_ty=T.u32)))

        a24 = BinOp(T.u32, op=Arith.BAND, lhs=a,
                    rhs=Lit(T.u32, value=0xFFFFFF))
        b24 = BinOp(T.u32, op=Arith.BAND, lhs=b,
                    rhs=Lit(T.u32, value=0xFFFFFF))
        m24_name = ctx.kx_fresh_name("ii_m24")
        m24 = Var(T.u32, name=m24_name)
        ctx.kx_declare(m24)
        stmts.append(Declare(name=m24_name, gtype=T.u32,
                             init=_u(Call(T.i32, fn="__mul24", args=[a24, b24],
                                          result_ty=T.i32))))

        mnab_name = ctx.kx_fresh_name("ii_mnab")
        mncd_name = ctx.kx_fresh_name("ii_mncd")
        mxmm_name = ctx.kx_fresh_name("ii_mxmm")
        mnab = Var(T.u32, name=mnab_name)
        mncd = Var(T.u32, name=mncd_name)
        mxmm = Var(T.u32, name=mxmm_name)
        for v in (mnab, mncd, mxmm):
            ctx.kx_declare(v)
        stmts.append(Declare(name=mnab_name, gtype=T.u32,
                             init=Call(T.u32, fn="min", args=[a, b],
                                       result_ty=T.u32)))
        stmts.append(Declare(name=mncd_name, gtype=T.u32,
                             init=Call(T.u32, fn="min", args=[c, d],
                                       result_ty=T.u32)))
        stmts.append(Declare(name=mxmm_name, gtype=T.u32,
                             init=Call(T.u32, fn="max", args=[mnab, mncd],
                                       result_ty=T.u32)))

        stmts.append(Assign(acc, _kx_xor(acc, _kx_xor(sad, _kx_xor(m24, mxmm)))))
        return stmts

    # ------------------------------------------------------------------
    # 6. Half precision (sm_53+ / CUDA only)
    # ------------------------------------------------------------------

    def _build_half_ops(self, ctx: GenContext, acc: Var, tid: Var, n_u: Expr,
                      input_var: Var, kx_is_cuda: bool, sm: int) -> List[Stmt]:
        """__float2half_rn/__hadd/__hmul round-trips (F2H / H2F / HADD / HMUL;
        sm_53+ only -- the dispatcher enforces the gate)."""
        stmts: List[Stmt] = []
        hf_name = ctx.kx_fresh_name("hp_hf")
        hf = Var(T.f32, name=hf_name)
        ctx.kx_declare(hf)
        idx_h = _mod(_kx_add(tid, Lit(T.u32, value=6)), n_u)
        # Scale to a small positive range to avoid half overflow (>65504).
        raw_h = BinOp(T.u32, op=Arith.BAND,
                      lhs=Subscript(T.u32, array=input_var, index=idx_h),
                      rhs=Lit(T.u32, value=0x3FFF))
        stmts.append(Declare(name=hf_name, gtype=T.f32, init=_f32(raw_h)))

        ha_name = ctx.kx_fresh_name("hp_ha")
        hb_name = ctx.kx_fresh_name("hp_hb")
        ha = Var(T.f16, name=ha_name)
        hb = Var(T.f16, name=hb_name)
        ctx.kx_declare(ha)
        ctx.kx_declare(hb)
        stmts.append(Declare(name=ha_name, gtype=T.f16,
                             init=Call(T.f16, fn="__float2half_rn",
                                       args=[hf], result_ty=T.f16)))
        # Second operand: slightly perturbed by the thread id.
        hf_perturb = BinOp(T.f32, op=Arith.ADD, lhs=hf,
                           rhs=_f32(BinOp(T.u32, op=Arith.BAND, lhs=tid,
                                          rhs=Lit(T.u32, value=0xF))))
        stmts.append(Declare(name=hb_name, gtype=T.f16,
                             init=Call(T.f16, fn="__float2half_rn",
                                       args=[hf_perturb], result_ty=T.f16)))

        hadd_name = ctx.kx_fresh_name("hp_hadd")
        hmul_name = ctx.kx_fresh_name("hp_hmul")
        hadd = Var(T.f16, name=hadd_name)
        hmul = Var(T.f16, name=hmul_name)
        ctx.kx_declare(hadd)
        ctx.kx_declare(hmul)
        stmts.append(Declare(name=hadd_name, gtype=T.f16,
                             init=Call(T.f16, fn="__hadd", args=[ha, hb],
                                       result_ty=T.f16)))
        stmts.append(Declare(name=hmul_name, gtype=T.f16,
                             init=Call(T.f16, fn="__hmul", args=[ha, hb],
                                       result_ty=T.f16)))

        hadd_f_name = ctx.kx_fresh_name("hp_hadd_f")
        hmul_f_name = ctx.kx_fresh_name("hp_hmul_f")
        hadd_f = Var(T.f32, name=hadd_f_name)
        hmul_f = Var(T.f32, name=hmul_f_name)
        ctx.kx_declare(hadd_f)
        ctx.kx_declare(hmul_f)
        stmts.append(Declare(name=hadd_f_name, gtype=T.f32,
                             init=Call(T.f32, fn="__half2float", args=[hadd],
                                       result_ty=T.f32)))
        stmts.append(Declare(name=hmul_f_name, gtype=T.f32,
                             init=Call(T.f32, fn="__half2float", args=[hmul],
                                       result_ty=T.f32)))

        stmts.append(Assign(acc, _kx_xor(acc, _kx_xor(_f_as_i(hadd_f),
                                                _f_as_i(hmul_f)))))
        return stmts

    # ------------------------------------------------------------------
    # 7. Local array access
    # ------------------------------------------------------------------

    def _build_local_array(self, ctx: GenContext, acc: Var, tid: Var,
                         n_u: Expr, input_var: Var, kx_is_cuda: bool,
                         sm: int) -> List[Stmt]:
        """Small thread-private array with computed indices (local-memory
        allocation and dynamic-index addressing).  Emitted as RawCode because
        the IR has no first-class stack-array lvalue; the tracked Var carries
        TArray so the checksum fold skips the array itself -- its contents
        still reach the accumulator via the la_rd/la_rd2 read-backs."""
        arr_size = 8
        stmts: List[Stmt] = []

        arr_name = ctx.kx_fresh_name("la_arr")
        ctx.kx_declare(Var(TArray(T.u32, arr_size), name=arr_name))
        stmts.append(RawCode(text=f"unsigned int {arr_name}[{arr_size}];"))

        i_name = ctx.kx_fresh_name("la_i")
        input_name = input_var.name
        n_name = "n"  # fixed kernel parameter name (see gen/assembler.py)
        stmts.append(For(var=i_name, start=_i(0), end=_i(arr_size),
                         step=_i(1),
                         body=Block([RawCode(text=(
                             f"{arr_name}[{i_name}] = "
                             f"{input_name}[(tid + (unsigned int){i_name} "
                             f"* 7u + 1u) % (unsigned int){n_name}] "
                             f"^ (tid * ((unsigned int){i_name} + 1u));"))])))
        stmts.append(RawCode(
            text=f"{arr_name}[0] ^= (tid * 0x6C62272Eu + 1u);"))

        # Read back through a dynamically computed index to force dynamic
        # local-memory addressing.
        dyn_name = ctx.kx_fresh_name("la_didx")
        rd_name = ctx.kx_fresh_name("la_rd")
        rd2_name = ctx.kx_fresh_name("la_rd2")
        ctx.kx_declare(Var(T.u32, name=rd_name))
        stmts.append(RawCode(text=(f"unsigned int {dyn_name} = "
                                   f"(tid * 0x6Bu + 5u) % {arr_size}u;")))
        stmts.append(RawCode(
            text=f"unsigned int {rd_name} = {arr_name}[{dyn_name}];"))
        stmts.append(RawCode(
            text=(f"unsigned int {rd2_name} = "
                  f"{arr_name}[({dyn_name} + 3u) % {arr_size}u];")))
        stmts.append(RawCode(
            text=f"{acc.name} = {acc.name} ^ {rd_name} ^ {rd2_name};"))
        return stmts

    # ------------------------------------------------------------------
    # 8. Register spill
    # ------------------------------------------------------------------

    def _build_register_spill(self, ctx: GenContext, acc: Var, tid: Var,
                            n_u: Expr, input_var: Var, kx_is_cuda: bool,
                            sm: int) -> List[Stmt]:
        """16 cross-mixed thread-local scalars pressuring the register
        allocator's spill/fill heuristics."""
        n_regs = 16
        stmts: List[Stmt] = []
        regs: List[Var] = []
        for i in range(n_regs):
            vname = ctx.kx_fresh_name("rs_v")
            v = Var(T.u32, name=vname)
            ctx.kx_declare(v)
            regs.append(v)
            idx = _mod(_kx_add(tid, Lit(T.u32, value=i * 3 + 1)), n_u)
            stmts.append(Declare(name=vname, gtype=T.u32,
                init=_kx_xor(Subscript(T.u32, array=input_var, index=idx),
                          Lit(T.u32,
                              value=(0x9E3779B9 * (i + 1)) & 0xFFFFFFFF))))

        # Cross-mix pass 1: v[i] ^= v[(i+3)%N] + v[(i+7)%N]
        for i in range(n_regs):
            j = (i + 3) % n_regs
            k = (i + 7) % n_regs
            stmts.append(Assign(regs[i],
                                _kx_xor(regs[i], _kx_add(regs[j], regs[k]))))

        # Cross-mix pass 2: v[i] += v[(i+5)%N] ^ (tid * (i+1))
        for i in range(n_regs):
            j = (i + 5) % n_regs
            noise = _mul(tid, Lit(T.u32, value=(i + 1) & 0xFFFF))
            stmts.append(Assign(regs[i],
                                _kx_add(regs[i], _kx_xor(regs[j], noise))))

        fold: Expr = regs[0]
        for v in regs[1:]:
            fold = _kx_xor(fold, v)
        stmts.append(Assign(acc, _kx_xor(acc, fold)))
        return stmts
