"""Random expression generator.

``ExprGen.kx_generate(target_type, depth)`` builds a typed expression tree via a
swarm-weighted dispatch over expression kinds.  Safety policy:

  * div/mod and shifts route through SafeBin (UB-free) unless the
    raw-division diversity knob is on, in which case divisors are explicitly
    guarded non-zero;
  * in observable/weave mode the pool is unsigned-only so the checksum path
    is bit-exact across compiler versions;
  * floating point is only generated when the type pool includes floats
    (p_float_types / allowed_types); integer operator paths never see an FP
    operand.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import List, Optional, Tuple

from generator_ir.ir.typedecl import (
    GType, T, TBool, TChar, TSChar, TShort, TUShort, TInt, TUInt, TLong,
    TULong, TFloat, TDouble, THalf, TBF16, TVec, TArray, TPointer,
    TQualified, Uniformity,
)
from generator_ir.ir.operators import Arith, Cmp, Logic, UnaOp, SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, Fma, SafeBin, Subscript, Call, WarpVote, AddrOf,
)
from generator_ir.ir.arch import Arch
from generator_ir.gen.context import GenContext
from generator_ir.ir import typing as type_ops


# (name, arg_type, result_type) -- includes the *ll 64-bit variants; calls
# are built at the intrinsic's natural types and 50% chained (perffuzzer).
_INT_INTRINSICS = (
    ("__clz", T.u32, T.i32), ("__popc", T.u32, T.i32),
    ("__ffs", T.u32, T.i32), ("__brev", T.u32, T.u32),
    ("__clzll", T.u64, T.i32), ("__popcll", T.u64, T.i32),
    ("__ffsll", T.u64, T.i32), ("__brevll", T.u64, T.u64),
)

# Curated per-type constant pools (yarpgen-style magic values that stress
# constant folding; perffuzzer expr_gen.py).
_UINT_CONSTANTS = [0, 1, 2, 3, 7, 13, 31, 37, 42, 0xFF, 0xDEAD, 0xBEEF,
                   0x9E3779B9, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF]
_INT_CONSTANTS = [0, 1, -1, 2, -2, 7, 13, 42, 127, -128, 255]
_SCHAR_CONSTANTS = [-128, -74, -1, 0, 1, 42, 63, 127]
_CHAR_CONSTANTS = [0, 1, 42, 50, 127, 200, 255]
_SHORT_CONSTANTS = [-32768, -1024, -1, 0, 1, 42, 1024, 13569, 32767]
_USHORT_CONSTANTS = [0, 1, 42, 1024, 33910, 49152, 65535]
_LONGLONG_CONSTANTS = [0, 1, -1, 42, 1638931535386871238,
                       -7293850184320197, 4611686018427387903]
_ULONGLONG_CONSTANTS = [0, 1, 42, 4986325391324574397,
                        9223372036854775807, 18446744073709551615]
_FLOAT_CONSTANTS = [0.0, 1.0, -1.0, 1.5, 2.0, 3.25, 7.0, 13.0, 31.0,
                    42.0, 0.125]
_INT_POOLS = {
    TChar: _CHAR_CONSTANTS, TSChar: _SCHAR_CONSTANTS,
    TShort: _SHORT_CONSTANTS, TUShort: _USHORT_CONSTANTS,
    TInt: _INT_CONSTANTS, TUInt: _UINT_CONSTANTS,
    TLong: _LONGLONG_CONSTANTS, TULong: _ULONGLONG_CONSTANTS,
}

# Catalog-driven intrinsics (ir/intrinsics.CATALOG).  Only categories that are
# integer-returning, deterministic, and take value/const_small operands are
# allowed into random expressions, so they fold safely into the cross-compiler
# integer checksum.  Deliberately EXCLUDED: warp_reduce/warp_vote/warp_match/
# warp_shuffle/sync (bare-full-mask collectives DEADLOCK on a non-converged
# warp), all FP/half/bf16/double (break the integer oracle), atomics/cache
# (need pointers), and void/clock (not values / nondeterministic).
_CATALOG_EXPR_CATEGORIES = frozenset({
    "bit_manip", "int_math", "longlong_math", "dpx",
})

# Dot-product intrinsics have overlapping signed/unsigned/packed overloads that
# scalar (uint,uint,int) arguments cannot disambiguate, so they are excluded.
# The SIGNED multiply-high / 24-bit-multiply intrinsics (__mul64hi/__mulhi/
# __mul24) produce DIFFERENT results on nvcc 11.5 vs 13.1 for negative operands
# -- a real cross-version divergence that would break the deterministic-checksum
# oracle's premise.  Their UNSIGNED counterparts are deterministic and kept.
_CATALOG_EXPR_NAME_BLOCK = frozenset({
    "__dp4a", "__dp2a_lo", "__dp2a_hi",
    "__mul64hi", "__mulhi", "__mul24",
})


@lru_cache(maxsize=None)
def _kx_expr_safe_intrinsics(arch_version: int) -> tuple:
    """Catalog intrinsics safe to drop into a random integer expression.

    Cached per arch.  Entries from ir/intrinsics.CATALOG filtered to the
    expression-safe categories and gated to the target SM version.
    """
    from generator_ir.ir.intrinsics import CATALOG
    return tuple(
        intr for intr in CATALOG.available(arch_version)
        if intr.category in _CATALOG_EXPR_CATEGORIES
        and intr.name not in _CATALOG_EXPR_NAME_BLOCK
    )


class ExprGen:
    """Depth-bounded, type-directed random expression builder."""

    def __init__(self, ctx: GenContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------
    # small type helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base(t: GType) -> GType:
        while isinstance(t, (TQualified,)):
            t = t.unqualified()
        return t

    @staticmethod
    def _is_vec(t: GType) -> bool:
        return isinstance(ExprGen._base(t), TVec)

    @staticmethod
    def _is_fp(t: GType) -> bool:
        return ExprGen._base(t).is_fp

    @staticmethod
    def _is_int(t: GType) -> bool:
        return ExprGen._base(t).is_int

    def _arch(self) -> Arch:
        return Arch.parse(self.ctx.config.isa)

    def _kx_is_cuda(self) -> bool:
        return not self.ctx.is_hip()

    def _sm_at_least(self, v: int) -> bool:
        a = self._arch()
        return a.kx_is_nvidia and a.version >= v

    def _in_weave(self) -> bool:
        return bool(getattr(self.ctx, "_weave_mode", False))

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------

    def kx_generate(self, target_type: GType, depth: int = 0) -> Expr:
        target_type = self._base(target_type)
        if depth >= self.ctx.config.expr_tree_limit:
            return self._kx_terminal(target_type)

        kinds = self._kind_table(target_type, depth)
        if not kinds:
            return self._kx_terminal(target_type)
        # Swarm perturbation (omit a random subset, jitter the rest);
        # "terminal" is protected so the tree can always bottom out.
        kinds = self.ctx.weights.apply("expr", kinds, protected=("terminal",))
        names = [k for k, _ in kinds]
        wts = [w for _, w in kinds]
        kind = self.ctx.rng.choices(names, weights=wts, k=1)[0]
        return self._kx_dispatch(kind, target_type, depth)

    def _kind_table(self, t: GType, depth: int) -> List[Tuple[str, float]]:
        """(kind, base weight) for a target type at a recursion depth."""
        table: List[Tuple[str, float]] = [("terminal", 2.0)]
        scalar_int = self._is_int(t) and not self._is_vec(t)
        scalar_fp = self._is_fp(t) and not self._is_vec(t)

        if scalar_int or scalar_fp:
            table += [
                ("binary", 3.0),
                ("cast", 1.0),
                ("cond_expr", 0.8),
                ("mixed_ternary", 0.5),
                ("cast_chain", 0.6),
            ]
        if scalar_int:
            table += [
                ("unary", 1.2),
                ("safe_bin", 1.5),
                ("shift_chain", 0.7),
                ("intrinsic", 0.8),
                ("mixed_binary", 0.6),
                ("const_xform", 0.4),
                ("fake_diverge", 0.3),
                ("group_diverge", 0.3),
                ("ldg_load", 0.3),
            ]
            if self._kx_is_cuda():
                # Data-driven intrinsic catalog (ir/intrinsics.CATALOG),
                # arch-gated; supersedes the hardcoded 8-entry list.
                table.append(("catalog_intrinsic", 0.6))
                table.append(("rounding_mode", 0.4))
                if self.ctx.config.enable_warp:
                    if self._sm_at_least(53):
                        table.append(("cuda_fp_bridge", 0.6))
                    if self._sm_at_least(80):
                        table.append(("cuda_cache_load", 0.4))
                    if self._sm_at_least(90):
                        table.append(("cuda_dpx", 0.3))
                if self._sm_at_least(53):
                    table.append(("half2_packed", 0.3))
            if self._in_weave() or self.ctx.config.observable_fold:
                p_u64 = max(0.0, float(getattr(
                    self.ctx.config, "prob_u64_divmod", 0.0)))
                if p_u64 > 0.0:
                    table.append(("u64_divmod", 1.2 * p_u64))
                table.append(("mul_wide", 0.4))
            if self._device_arrays_available():
                table.append(("device_array_read", 0.5))
            if self._helpers_available():
                table.append(("helper_call", 0.5))
            # observable_fold: any FP in the observed path leaks cross-compiler
            # float differences (FMA contraction, FP16/rounding-mode) into the
            # integer XOR checksum -> false mismatch.  Keep it integer-exact.
            if getattr(self.ctx.config, "observable_fold", False):
                drop = {"cuda_fp_bridge", "half2_packed", "rounding_mode"}
                table = [(k, w) for k, w in table if k not in drop]
        if scalar_fp:
            table += [
                ("fma", 1.2),
                ("fp_libcall", 0.6),
            ]
            if self._kx_is_cuda():
                if self._sm_at_least(53):
                    table.append(("cuda_fp_bridge", 0.8))
                if self._sm_at_least(80):
                    table.append(("cuda_fp_bridge", 0.4))
        if isinstance(self._base(t), TBool):
            table += [("comparison", 3.0), ("logic", 1.5)]
        # warp_vote exprs: suppressed on gfx1200 (warp-sync semantics are
        # problematic there; mirrors the original's is_gfx1200 gate).
        isa = getattr(self.ctx.config, "isa", "")
        gfx1200 = isa.startswith("gfx") and self.ctx.kx_arch_number() == 1200
        if (self.ctx.config.enable_warp and self._uniform_context()
                and self._arch().warp_sync and not gfx1200):
            table.append(("warp_vote", 0.4))
        return table

    def _kx_dispatch(self, kind: str, t: GType, depth: int) -> Expr:
        fn = {
            "terminal": self._kx_terminal,
            "binary": self._binary,
            "unary": self._unary,
            "cast": self._cast,
            "cond_expr": self._cond_expr,
            "comparison": self._comparison,
            "logic": self._logic,
            "safe_bin": self._safe_bin,
            "shift_chain": self._shift_chain,
            "intrinsic": self._intrinsic,
            "mixed_binary": self._mixed_binary,
            "mixed_ternary": self._mixed_ternary,
            "cast_chain": self._cast_chain,
            "u64_divmod": self._u64_divmod,
            "mul_wide": self._mul_wide,
            "fma": self._fma,
            "fp_libcall": self._fp_libcall,
            "const_xform": self._const_xform,
            "fake_diverge": self._fake_diverge,
            "group_diverge": self._group_diverge,
            "ldg_load": self._ldg_load,
            "catalog_intrinsic": self._catalog_intrinsic,
            "cuda_cache_load": self._cuda_cache_load,
            "cuda_fp_bridge": self._cuda_fp_bridge,
            "cuda_dpx": self._cuda_dpx,
            "half2_packed": self._half2_packed,
            "rounding_mode": self._rounding_mode,
            "device_array_read": self._device_array_read,
            "helper_call": self._helper_call,
            "warp_vote": self._warp_vote,
        }.get(kind, self._kx_terminal)
        return fn(t, depth)

    def _uniform_context(self) -> bool:
        c = self.ctx
        return (c.if_depth == 0
                and not c.inside_breakable_loop
                and not c.in_nested_control_flow
                and not c.in_device_helper)

    def _device_arrays_available(self) -> bool:
        return bool(getattr(self.ctx, "_device_arrays", None))

    def _helpers_available(self) -> bool:
        return bool(getattr(self.ctx, "_device_helpers", None))

    # ------------------------------------------------------------------
    # terminals
    # ------------------------------------------------------------------

    def _kx_terminal(self, t: GType, depth: int = 0) -> Expr:
        roll = self.ctx.rnd_float()
        if roll < 0.45:
            var = self._pick_var(t)
            if var is not None:
                return var
        if roll < 0.8:
            return self._kx_constant(t)
        return self._kx_thread_builtin(t)

    def _kx_constant(self, t: GType) -> Expr:
        t = self._base(t)
        wide = self.ctx.rnd_float() < getattr(self.ctx.config, "p_wide_constants", 0.0)
        if isinstance(t, TBool):
            return Lit(t, value=bool(self.ctx.rnd_int(0, 1)))
        if t.is_int:
            if wide:
                # Full-range immediate for this type: different values drive
                # different ptxas constant materialisation (ULDC / IMAD-chain
                # / PRMT) that a fixed pool keeps re-hitting identically.
                lo, hi = t.limits
                value = self.ctx.rng.randint(lo, hi)
            else:
                pool = _INT_POOLS.get(type(t), _UINT_CONSTANTS)
                value = self.ctx.rnd_choice(pool)
            return Lit(t, value=value)
        if t.is_fp:
            value = (self.ctx.rng.uniform(-1000.0, 1000.0) if wide
                     else self.ctx.rnd_choice(_FLOAT_CONSTANTS))
            return Lit(t, value=float(value))
        return Lit(T.u32, value=self.ctx.rnd_choice(_UINT_CONSTANTS))

    def _pick_var(self, t: GType) -> Optional[Expr]:
        matches = self.ctx.kx_scope.kx_variables_of_type(
            t, only_names=self.ctx._visible_names)
        if matches:
            return Var(t, name=self.ctx.rnd_choice(matches).name)
        return None

    def _kx_thread_builtin(self, t: GType) -> Expr:
        src = self.ctx.rnd_choice(
            ["threadIdx", "blockIdx", "blockDim", "gridDim"])
        axis = self.ctx.rnd_choice(["x", "x", "x", "y"])
        expr: Expr = ThreadBuiltin(T.u32, axis=axis, source=src,
                                   uniformity=(Uniformity.VARYING if src == "threadIdx"
                                               else Uniformity.UNIFORM))
        if self._is_int(t) and not isinstance(self._base(t), TUInt):
            return Cast(t, to=t, arg=expr)
        if self._is_fp(t):
            return Cast(t, to=t, arg=expr)
        return expr

    # ------------------------------------------------------------------
    # arithmetic kinds
    # ------------------------------------------------------------------

    @staticmethod
    def _signed_wrap_op(op: Arith) -> Optional[SafeOp]:
        return {Arith.ADD: SafeOp.SADD,
                Arith.SUB: SafeOp.SSUB,
                Arith.MUL: SafeOp.SMUL}.get(op)

    def _arith(self, op: Arith, lhs: Expr, rhs: Expr, t: GType) -> Expr:
        """Arithmetic node for (op, lhs, rhs) at type t.

        Signed +, -, * route through the unsigned-wraparound SafeOp macros so
        overflow is DEFINED behavior (two compilers cannot exploit signed
        overflow UB to produce divergent checksums).  Unsigned and bitwise
        ops are emitted raw (already UB-free)."""
        base = self._base(t)
        if base.kx_is_signed and base.is_int:
            safe = self._signed_wrap_op(op)
            if safe is not None:
                return SafeBin(base, op=safe, lhs=lhs, rhs=rhs)
        return BinOp(base, op=op, lhs=lhs, rhs=rhs)

    def _int_binops(self, t: GType) -> List[Arith]:
        ops = [Arith.ADD, Arith.SUB, Arith.MUL,
               Arith.BAND, Arith.BOR, Arith.BXOR]
        return ops

    def _binary(self, t: GType, depth: int) -> Expr:
        t = self._base(t)
        if self._is_fp(t):
            op = self.ctx.rnd_choice([Arith.ADD, Arith.SUB, Arith.MUL])
            return BinOp(t, op=op,
                         lhs=self.kx_generate(t, depth + 1),
                         rhs=self.kx_generate(t, depth + 1))
        op = self.ctx.rnd_choice(self._int_binops(t))
        lhs = self.kx_generate(t, depth + 1)
        rhs = self.kx_generate(t, depth + 1)
        return self._arith(op, lhs, rhs, t)

    def _mixed_binary(self, t: GType, depth: int) -> Expr:
        """Binary op with operands of two different type classes; the result
        type follows the usual arithmetic conversions (implicit promotions
        stress type-lowering and sign-extension passes)."""
        t = self._base(t)
        others = [x for x in self.ctx.kx_computation_types()
                  if type(x) is not type(t) and not isinstance(x, TBool)]
        if not others:
            return self._binary(t, depth)
        other = self.ctx.rnd_choice(others)
        lhs = self.kx_generate(t, depth + 1)
        rhs = self.kx_generate(other, depth + 1)
        if self._is_fp(t) or self._is_fp(other):
            op = self.ctx.rnd_choice([Arith.ADD, Arith.SUB, Arith.MUL])
        else:
            op = self.ctx.rnd_choice([Arith.ADD, Arith.SUB, Arith.BAND,
                                       Arith.BOR, Arith.BXOR])
        rty = type_ops.kx_arithmetic_result_type(op, t, other)
        node = self._arith(op, lhs, rhs, rty)
        if type(rty) is not type(t):
            node = Cast(t, to=t, arg=node, soft=True)
        return node

    def _unary(self, t: GType, depth: int) -> Expr:
        op = self.ctx.rnd_choice([UnaOp.BNOT, UnaOp.NEG])
        arg = self.kx_generate(t, depth + 1)
        base = self._base(t)
        if op is UnaOp.NEG and base.kx_is_signed and base.is_int:
            # -x on a signed int is UB at INT_MIN; emit 0 - x through the
            # unsigned-wrap macro instead (identical result, defined).
            zero = Lit(base, value=0)
            return SafeBin(base, op=SafeOp.SSUB, lhs=zero, rhs=arg)
        return UnOp(t, op=op, arg=arg)

    def _safe_bin(self, t: GType, depth: int) -> Expr:
        t = self._base(t)
        if not self._is_int(t):
            return self._binary(t, depth)
        raw_p = getattr(self.ctx.config, "p_raw_division", 0.0)
        if self.ctx.rnd_float() < raw_p:
            return self._raw_divmod(t, depth)
        op = self.ctx.rnd_choice(
            [SafeOp.SDIV, SafeOp.SMOD, SafeOp.SSHL, SafeOp.SSHR])
        lhs = self.kx_generate(t, depth + 1)
        rhs = self.kx_generate(t, depth + 1)
        return SafeBin(t, op=op, lhs=lhs, rhs=rhs)

    def _raw_divmod(self, t: GType, depth: int) -> Expr:
        """Bare / or % with a guarded divisor.  For SIGNED targets we cannot
        use the raw operator (INT_MIN / -1 overflows), so the operation is
        evaluated in the unsigned domain and cast back (defined wrap)."""
        t = self._base(t)
        op = self.ctx.rnd_choice([Arith.DIV, Arith.MOD])
        lhs = self.kx_generate(t, depth + 1)
        rhs_inner = self.kx_generate(t, depth + 1)
        one = Lit(t, value=1)
        divisor = BinOp(t, op=Arith.BOR, lhs=rhs_inner, rhs=one)
        if t.kx_is_signed:
            # unsigned evaluate: (T)((u32)a / (u32)(b|1))
            u = T.u32 if t.width <= 4 else T.u64
            udiv = BinOp(u, op=op,
                         lhs=Cast(u, to=u, arg=lhs),
                         rhs=Cast(u, to=u, arg=divisor))
            return Cast(t, to=t, arg=udiv)
        return BinOp(t, op=op, lhs=lhs, rhs=divisor)

    def _u64_divmod(self, t: GType, depth: int) -> Expr:
        """64-bit division/modulo by a runtime nonzero divisor (u64 stress)."""
        dividend = self.kx_generate(T.u64, depth + 1)
        divisor = BinOp(T.u64, op=Arith.BOR,
                        lhs=self.kx_generate(T.u64, depth + 1),
                        rhs=Lit(T.u64, value=1))
        op = self.ctx.rnd_choice([Arith.DIV, Arith.MOD])
        wide = BinOp(T.u64, op=op, lhs=dividend, rhs=divisor)
        return Cast(t, to=t, arg=wide)

    def _mul_wide(self, t: GType, depth: int) -> Expr:
        """Widening multiply through u64 (IMAD.HI surface), then narrow back."""
        lhs = Cast(T.u64, to=T.u64, arg=self.kx_generate(t, depth + 1))
        rhs = Cast(T.u64, to=T.u64, arg=self.kx_generate(t, depth + 1))
        wide = BinOp(T.u64, op=Arith.MUL, lhs=lhs, rhs=rhs)
        return Cast(t, to=t, arg=wide)

    def _shift_chain(self, t: GType, depth: int) -> Expr:
        """(a << (b & 31)) ^ (c >> (b & 31)) -- data-dependent shift amounts
        masked to the valid 0-31 range (perffuzzer pattern)."""
        t = self._base(t)
        a = self.kx_generate(T.u32, depth + 1)
        b = self.kx_generate(T.u32, depth + 1)
        c = self.kx_generate(T.u32, depth + 1)
        amt1 = BinOp(T.u32, op=Arith.BAND, lhs=b, rhs=Lit(T.u32, value=31))
        amt2 = BinOp(T.u32, op=Arith.BAND, lhs=copy.deepcopy(b),
                     rhs=Lit(T.u32, value=31))
        left = SafeBin(T.u32, op=SafeOp.SSHL, lhs=a, rhs=amt1)
        right = SafeBin(T.u32, op=SafeOp.SSHR, lhs=c, rhs=amt2)
        result = BinOp(T.u32, op=Arith.BXOR, lhs=left, rhs=right)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _cast(self, t: GType, depth: int) -> Expr:
        t = self._base(t)
        pool = [x for x in self.ctx.kx_computation_types() if type(x) is not type(t)]
        if not pool:
            return self._kx_terminal(t)
        src_t = self.ctx.rnd_choice(pool)
        inner = self.kx_generate(src_t, depth + 1)
        # ~1/3 of casts go implicit (stresses type-lowering passes)
        soft = self.ctx.rnd_bool(0.33)
        return Cast(t, to=t, arg=inner, soft=soft)

    def _cond_expr(self, t: GType, depth: int) -> Expr:
        cond = self._comparison(T.boolean, depth + 1)
        # Ternary arms are divergent contexts: bump if_depth (mirrors the
        # original) so uniform-gated kinds (warp_vote) stay out of the arms.
        self.ctx.if_depth += 1
        try:
            when_true = self.kx_generate(t, depth + 1)
            when_false = self.kx_generate(t, depth + 1)
        finally:
            self.ctx.if_depth -= 1
        return Select(t, cond=cond, when_true=when_true, when_false=when_false)

    def _mixed_ternary(self, t: GType, depth: int) -> Expr:
        """Ternary whose branches have different non-bool types; the result
        uses the common type (implicit conversions in both branches)."""
        t = self._base(t)
        non_bool = [x for x in self.ctx.kx_computation_types()
                    if not isinstance(x, TBool)]
        if len(non_bool) < 2:
            return self._cond_expr(t, depth)
        ta = self.ctx.rnd_choice(non_bool)
        rest = [x for x in non_bool if type(x) is not type(ta)]
        if not rest:
            return self._cond_expr(t, depth)
        tb = self.ctx.rnd_choice(rest)
        cond = self._comparison(T.boolean, depth + 1)
        rty = type_ops.kx_common_type(ta, tb)
        self.ctx.if_depth += 1
        try:
            when_true = self.kx_generate(ta, depth + 1)
            when_false = self.kx_generate(tb, depth + 1)
        finally:
            self.ctx.if_depth -= 1
        node = Select(rty, cond=cond,
                      when_true=when_true, when_false=when_false)
        if type(rty) is not type(t):
            node = Cast(t, to=t, arg=node, soft=True)
        return node

    def _cast_chain(self, t: GType, depth: int) -> Expr:
        """Multi-step cast chain through diverse types (promotion /
        sign-extension / truncation stress)."""
        t = self._base(t)
        pool = [x for x in self.ctx.kx_computation_types()
                if not isinstance(x, TBool)]
        if not pool:
            return self._kx_terminal(t)
        node = self.kx_generate(self.ctx.kx_primary_type(), depth + 1)
        steps = self.ctx.rnd_int(2, 3) if len(pool) > 1 else 1
        for _ in range(steps):
            mid = self.ctx.rnd_choice(pool)
            node = Cast(mid, to=mid, arg=node,
                        soft=self.ctx.rnd_bool(0.5))
        if not (isinstance(node, Cast) and type(node.to) is type(t)):
            node = Cast(t, to=t, arg=node)
        return node

    def _comparison(self, t: GType, depth: int) -> Expr:
        pool = [x for x in self.ctx.kx_computation_types()]
        operand_t = self.ctx.rnd_choice(pool)
        lhs = self.kx_generate(operand_t, depth + 1)
        rhs = self.kx_generate(operand_t, depth + 1)
        op = self.ctx.rnd_choice(list(Cmp))
        return RelOp(T.boolean, op=op, lhs=lhs, rhs=rhs)

    def _logic(self, t: GType, depth: int) -> Expr:
        op = self.ctx.rnd_choice(list(Logic))
        return LogicOp(T.boolean, op=op,
                       lhs=self._comparison(T.boolean, depth + 1),
                       rhs=self._comparison(T.boolean, depth + 1))

    def _intrinsic(self, t: GType, depth: int) -> Expr:
        """Integer intrinsic call (__clz/__popc/__ffs/__brev + *ll variants);
        50% chained via XOR for a data dependency (perffuzzer pattern)."""
        t = self._base(t)
        if self.ctx.rnd_float() < 0.2:
            # gpuperf extra: __byte_perm permutes two u32s by a selector.
            args = [self.kx_generate(T.u32, depth + 1),
                    self.kx_generate(T.u32, depth + 1),
                    Lit(T.u32, value=self.ctx.rnd_int(0, 7))]
            result: Expr = Call(T.u32, fn="__byte_perm", args=args,
                                result_ty=T.u32)
        else:
            name, aty, rty = self.ctx.rnd_choice(_INT_INTRINSICS)
            arg = Cast(aty, to=aty, arg=self.kx_generate(aty, depth + 1))
            result = Cast(T.u32, to=T.u32,
                          arg=Call(rty, fn=name, args=[arg], result_ty=rty))
            if self.ctx.rnd_bool(0.5):
                name2, aty2, rty2 = self.ctx.rnd_choice(_INT_INTRINSICS)
                arg2 = Cast(aty2, to=aty2, arg=self.kx_generate(aty2, depth + 1))
                c2 = Cast(T.u32, to=T.u32,
                          arg=Call(rty2, fn=name2, args=[arg2],
                                   result_ty=rty2))
                result = BinOp(T.u32, op=Arith.BXOR, lhs=result, rhs=c2)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _const_xform(self, t: GType, depth: int) -> Expr:
        """A constant modified by a transform (XOR mask / bitwise NOT / mixed
        with a variable) -- values the compiler cannot trivially fold
        (constant-folding / algebraic-simplification stress)."""
        t = self._base(t)
        if self._is_fp(t) or self._is_vec(t):
            return self._kx_terminal(t)
        base = self._kx_constant(T.u32)
        xform = self.ctx.rnd_choice(["xor", "not", "add_var"])
        if xform == "xor":
            mask = Lit(T.u32, value=self.ctx.rnd_choice(
                [0x55555555, 0xAAAAAAAA, 0xFF00FF00, 0x0F0F0F0F]))
            result: Expr = BinOp(T.u32, op=Arith.BXOR, lhs=base, rhs=mask)
        elif xform == "not":
            result = UnOp(T.u32, op=UnaOp.BNOT, arg=base)
        else:
            var = self.kx_generate(T.u32, depth + 1)
            result = BinOp(T.u32, op=Arith.BXOR, lhs=base, rhs=var)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _fake_diverge(self, t: GType, depth: int) -> Expr:
        """GPU_FAKE_DIVERGE(threadIdx.x, threadIdx.x) -- looks divergent to
        the compiler but is always 0 (divergence-analysis stress); mixed
        into the tree via addition."""
        t = self._base(t)
        tid = ThreadBuiltin(T.u32, axis="x", source="threadIdx",
                            uniformity=Uniformity.VARYING)
        fake = Call(T.u32, fn="GPU_FAKE_DIVERGE",
                    args=[tid, ThreadBuiltin(T.u32, axis="x", source="threadIdx",
                                             uniformity=Uniformity.VARYING)],
                    result_ty=T.u32)
        inner = self.kx_generate(T.u32, depth + 1)
        result = BinOp(T.u32, op=Arith.ADD, lhs=inner, rhs=fake)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _group_diverge(self, t: GType, depth: int) -> Expr:
        """GPU_GROUP_DIVERGE(blockIdx.x, val) == val -- uniform within a
        block (cross-block uniformity-analysis stress)."""
        t = self._base(t)
        bid = ThreadBuiltin(T.u32, axis="x", source="blockIdx",
                            uniformity=Uniformity.UNIFORM)
        val = self.kx_generate(T.u32, depth + 1)
        result = Call(T.u32, fn="GPU_GROUP_DIVERGE", args=[bid, val],
                      result_ty=T.u32)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    # ------------------------------------------------------------------
    # read-only / cache-hint loads
    # ------------------------------------------------------------------

    def _bounded_input_index(self) -> Optional[Expr]:
        """tid %% n (or tid & 0xFFFF without n) for in-bounds input reads."""
        if (self.ctx.kx_scope.lookup("input") is None
                or self.ctx.kx_scope.lookup("tid") is None):
            return None
        tid = Var(T.u32, name="tid")
        if self.ctx.kx_scope.lookup("n") is not None:
            return SafeBin(T.u32, op=SafeOp.SMOD, lhs=tid,
                           rhs=Cast(T.u32, to=T.u32, arg=Var(T.i32, name="n")))
        return BinOp(T.u32, op=Arith.BAND, lhs=tid, rhs=Lit(T.u32, value=0xFFFF))

    def _ldg_load(self, t: GType, depth: int) -> Expr:
        """__ldg(&input[idx]) -- read-only (texture) cache path load."""
        t = self._base(t)
        idx = self._bounded_input_index()
        if idx is None:
            return self._kx_terminal(t)
        addr = AddrOf(TPointer(T.u32),
                      arg=Subscript(T.u32,
                                    array=Var(TPointer(T.u32), name="input"),
                                    index=idx))
        result = Call(T.u32, fn="__ldg", args=[addr], result_ty=T.u32)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _cuda_cache_load(self, t: GType, depth: int) -> Expr:
        """CUDA cache-hint loads (__ldg/__ldcg/__ldca/__ldcs/__ldlu), 35%
        XOR-chained with a second hint for L1-policy interaction stress."""
        t = self._base(t)
        if not (self._kx_is_cuda() and self._sm_at_least(70)):
            return self._kx_terminal(t)
        idx = self._bounded_input_index()
        if idx is None:
            return self._kx_terminal(t)
        addr = AddrOf(TPointer(T.u32),
                      arg=Subscript(T.u32,
                                    array=Var(TPointer(T.u32), name="input"),
                                    index=idx))
        fn = self.ctx.rnd_choice(["__ldg", "__ldcg", "__ldca", "__ldcs"])
        if fn == "__ldg" and self._sm_at_least(80) and self.ctx.rnd_bool(0.3):
            fn = "__ldlu"
        result: Expr = Call(T.u32, fn=fn, args=[addr], result_ty=T.u32)
        if self.ctx.rnd_bool(0.35):
            alt_fn = self.ctx.rnd_choice(["__ldg", "__ldcg", "__ldca", "__ldcs"])
            if (alt_fn == "__ldg" and self._sm_at_least(80)
                    and self.ctx.rnd_bool(0.5)):
                alt_fn = "__ldlu"
            alt = Call(T.u32, fn=alt_fn, args=[addr], result_ty=T.u32)
            result = BinOp(T.u32, op=Arith.BXOR, lhs=result, rhs=alt)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    # ------------------------------------------------------------------
    # catalog-driven intrinsics
    # ------------------------------------------------------------------

    def _catalog_intrinsic(self, t: GType, depth: int) -> Expr:
        """A call to a catalog intrinsic (ir/intrinsics.CATALOG), arch-gated.

        Picks a deterministic integer/bit/dpx intrinsic available on the
        target SM and builds its arguments compositionally from the
        catalog's parameter roles, so operands are fuzzed sub-expressions.
        """
        t = self._base(t)
        if self._is_fp(t) or self._is_vec(t) or not self._kx_is_cuda():
            return self._kx_terminal(t)
        candidates = _kx_expr_safe_intrinsics(self._arch().version)
        if not candidates:
            return self._kx_terminal(t)
        intr = self.ctx.rnd_choice(list(candidates))
        args = [self._kx_catalog_arg(pt, role, depth)
                for pt, role in zip(intr.params, intr.roles)]
        result = Call(intr.result, fn=intr.name, args=args,
                      result_ty=intr.result)
        if type(self._base(intr.result)) is not type(t):
            return Cast(t, to=t, arg=result)
        return result

    def _kx_catalog_arg(self, ptype: GType, kind: str, depth: int) -> Expr:
        """One intrinsic argument per its catalog role; every argument is
        wrapped in an explicit cast so overload resolution is unambiguous."""
        if kind == "const_small":
            inner: Expr = Lit(ptype, value=self.ctx.rnd_choice(list(range(32))))
        elif kind == "mask":
            ptype = T.u32
            inner = Lit(ptype, value=0xFFFFFFFF)
        elif kind == "predicate":
            ptype = T.i32
            inner = self.kx_generate(ptype, depth + 1)
        else:  # "value" (and any unexpected role): a fuzzed sub-expression.
            inner = self.kx_generate(ptype, depth + 1)
        return Cast(ptype, to=ptype, arg=inner)

    # ------------------------------------------------------------------
    # CUDA FP/dpx surface
    # ------------------------------------------------------------------

    def _cuda_fp_bridge(self, t: GType, depth: int) -> Expr:
        """CUDA FP16/BF16 conversion and math chains (fp16 pipeline stress)."""
        t = self._base(t)
        if self._sm_at_least(80) and self.ctx.rnd_bool(0.5):
            src = self._binary(T.f32, depth + 1)
            half = Call(T.f16, fn="__float2half_rn", args=[src],
                        result_ty=T.f16)
            half_rhs = Call(T.f16, fn="__float2half_rn",
                            args=[self._binary(T.f32, depth + 1)],
                            result_ty=T.f16)
            half_math = Call(T.f16, fn="__hfma", args=[half, half_rhs, half_rhs],
                             result_ty=T.f16)
            back: Expr = Call(T.f32, fn="__half2float", args=[half_math],
                              result_ty=T.f32)
            if self.ctx.rnd_bool(0.35):
                back = Call(T.f32, fn="__fmaf_rn",
                            args=[back, self._binary(T.f32, depth + 1),
                                  self._binary(T.f32, depth + 1)],
                            result_ty=T.f32)
        elif self._sm_at_least(80):
            src = self._binary(T.f32, depth + 1)
            bf = Call(T.bf16, fn="__float2bfloat16_rn", args=[src],
                      result_ty=T.bf16)
            bf_rhs = Call(T.bf16, fn="__float2bfloat16_rn",
                          args=[self._binary(T.f32, depth + 1)],
                          result_ty=T.bf16)
            bf_math = Call(T.bf16, fn="__hfma", args=[bf, bf_rhs, bf_rhs],
                           result_ty=T.bf16)
            back = Call(T.f32, fn="__bfloat162float", args=[bf_math],
                        result_ty=T.f32)
        elif self._sm_at_least(53):
            src = self._binary(T.f32, depth + 1)
            half = Call(T.f16, fn="__float2half_rn", args=[src],
                        result_ty=T.f16)
            back = Call(T.f32, fn="__half2float", args=[half],
                        result_ty=T.f32)
        else:
            return self._kx_terminal(t)
        if isinstance(t, TFloat):
            return back
        # Bridge the FP result into integer contexts via __float_as_uint, the
        # deterministic bit move.  A numeric Cast of a half-overflow Inf/NaN
        # (emitted as ((unsigned int)(back))) is implementation-defined and can
        # diverge across compilers, corrupting the determinism oracle.  Mirrors
        # _rounding_mode / _half2_packed.
        as_uint: Expr = Call(T.u32, fn="__float_as_uint", args=[back],
                             result_ty=T.u32)
        if isinstance(t, TUInt):
            return as_uint
        return Cast(t, to=t, arg=as_uint)

    def _cuda_dpx(self, t: GType, depth: int) -> Expr:
        """DPX min/max/add instructions (sm_90+); 40% XOR-chained with an
        integer intrinsic for a data dependency."""
        t = self._base(t)
        if not (self._kx_is_cuda() and self._sm_at_least(90)):
            return self._kx_terminal(t)
        # CUDA 13 removed the plain 2-arg forms (__vimax_s32/__vimin_*);
        # keep only the 3-arg and _relu spellings that the whole compiler
        # fleet still declares.
        fn = self.ctx.rnd_choice([
            "__vimax3_s32", "__vimin3_s32", "__viaddmax_s32", "__viaddmin_s32",
            "__vimax_s32_relu", "__vimin_s32_relu",
            "__vimax3_u32", "__vimin3_u32", "__viaddmax_u32", "__viaddmin_u32",
        ])
        ty = T.u32 if "_u32" in fn else T.i32
        a = self._binary(ty, depth + 1)
        b = self._binary(ty, depth + 1)
        # Arity: vimax3/vimin3/viadd* take 3 operands; the 2-arg _relu
        # forms take 2 (note "s32" contains "3" -- match on "max3"/"min3").
        c = (self._binary(ty, depth + 1)
             if "max3" in fn or "min3" in fn or "add" in fn else None)
        args = [a, b] if c is None else [a, b, c]
        result: Expr = Call(ty, fn=fn, args=args, result_ty=ty)
        if self.ctx.rnd_bool(0.4):
            result = BinOp(ty, op=Arith.BXOR, lhs=result,
                           rhs=self._intrinsic(ty, depth + 1))
        if type(t) is ty:
            return result
        return Cast(t, to=t, arg=result)

    def _half2_packed(self, t: GType, depth: int) -> Expr:
        """Packed half2 arithmetic (sm_53+, CUDA only): __halves2half2 pairs,
        __hadd2/__hmul2, extract low half, reinterpret as uint.  half2 is not
        in the IR type system -- the packed calls are typed u32 and the
        emitter renders them verbatim (perffuzzer convention)."""
        t = self._base(t)
        if not (self._kx_is_cuda() and self._sm_at_least(53)):
            return self._kx_terminal(t)
        sub = max(0, min(depth + 2, self.ctx.config.expr_tree_limit - 1))
        halves = [Call(T.f16, fn="__float2half_rn",
                       args=[self.kx_generate(T.f32, sub)], result_ty=T.f16)
                  for _ in range(4)]
        pack1 = Call(T.u32, fn="__halves2half2", args=[halves[0], halves[1]],
                     result_ty=T.u32)
        pack2 = Call(T.u32, fn="__halves2half2", args=[halves[2], halves[3]],
                     result_ty=T.u32)
        op = self.ctx.rnd_choice(["__hadd2", "__hmul2"])
        packed = Call(T.u32, fn=op, args=[pack1, pack2], result_ty=T.u32)
        low = Call(T.f16, fn="__low2half", args=[packed], result_ty=T.f16)
        as_float = Call(T.f32, fn="__half2float", args=[low], result_ty=T.f32)
        result: Expr = Call(T.u32, fn="__float_as_uint", args=[as_float],
                            result_ty=T.u32)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _rounding_mode(self, t: GType, depth: int) -> Expr:
        """Float arithmetic with explicit rounding modes (CUDA only):
        __fadd_/__fmul_/__fmaf_/__fdiv_/__fsqrt_{rn,rd,ru,rz} -- each maps to
        a distinct PTX instruction variant.  Wrapped via __float_as_uint so
        it folds into integer contexts."""
        t = self._base(t)
        if not self._kx_is_cuda():
            return self._kx_terminal(t)
        sub = max(0, min(depth + 2, self.ctx.config.expr_tree_limit - 1))
        a = self.kx_generate(T.f32, sub)
        b = self.kx_generate(T.f32, sub)
        rm = self.ctx.rnd_choice(["rn", "rd", "ru", "rz"])
        op_type = self.ctx.rnd_choice(["add", "mul", "fma", "div", "sqrt"])
        if op_type == "add":
            fp = Call(T.f32, fn=f"__fadd_{rm}", args=[a, b], result_ty=T.f32)
        elif op_type == "mul":
            fp = Call(T.f32, fn=f"__fmul_{rm}", args=[a, b], result_ty=T.f32)
        elif op_type == "fma":
            c = self.kx_generate(T.f32, sub)
            fp = Call(T.f32, fn=f"__fmaf_{rm}", args=[a, b, c],
                      result_ty=T.f32)
        elif op_type == "div":
            fp = Call(T.f32, fn=f"__fdiv_{rm}", args=[a, b], result_ty=T.f32)
        else:
            fp = Call(T.f32, fn=f"__fsqrt_{rm}", args=[a], result_ty=T.f32)
        result: Expr = Call(T.u32, fn="__float_as_uint", args=[fp],
                            result_ty=T.u32)
        if isinstance(t, TUInt):
            return result
        return Cast(t, to=t, arg=result)

    def _fma(self, t: GType, depth: int) -> Expr:
        return Fma(t, a=self.kx_generate(t, depth + 1),
                   b=self.kx_generate(t, depth + 1),
                   c=self.kx_generate(t, depth + 1))

    def _fp_libcall(self, t: GType, depth: int) -> Expr:
        t = self._base(t)
        if isinstance(t, TDouble):
            fn = self.ctx.rnd_choice(["sin", "cos", "exp", "log", "sqrt"])
        else:
            fn = self.ctx.rnd_choice(
                ["sinf", "cosf", "expf", "logf", "sqrtf", "__logf", "__expf"])
        arg = self.kx_generate(t, depth + 1)
        if fn in ("logf", "sqrtf", "__logf", "log", "sqrt"):
            # keep domains valid: |x| + 1
            arg = BinOp(t, op=Arith.ADD,
                        lhs=Call(t, fn="fabsf" if isinstance(t, TFloat) else "fabs",
                                 args=[arg], result_ty=t),
                        rhs=Lit(t, value=1.0 if isinstance(t, TFloat) else 1.0))
        return Call(t, fn=fn, args=[arg], result_ty=t)

    def _device_array_read(self, t: GType, depth: int) -> Expr:
        arrays = getattr(self.ctx, "_device_arrays", None)
        if not arrays:
            return self._kx_terminal(t)
        info = self.ctx.rnd_choice(arrays)
        arr_var = Var(info["type"], name=info["name"])
        node: Expr = arr_var
        rem: GType = info["type"]
        for dim in info["dims"]:
            idx = SafeBin(T.u32, op=SafeOp.SMOD,
                          lhs=self.kx_generate(T.u32, depth + 1),
                          rhs=Lit(T.u32, value=dim))
            nxt = rem.elem if isinstance(rem, TArray) else T.u32
            node = Subscript(nxt, array=node, index=idx)
            rem = nxt
        if type(self._base(t)) is type(self._base(rem)):
            return node
        return Cast(t, to=t, arg=node)

    def _helper_call(self, t: GType, depth: int) -> Expr:
        helpers = getattr(self.ctx, "_device_helpers", None)
        if not helpers:
            return self._kx_terminal(t)
        h = self.ctx.rnd_choice(helpers)
        args = []
        for p in h.params:
            pt = self._base(p.gtype)
            args.append(Cast(pt, to=pt, arg=self.kx_generate(T.u32, depth + 1))
                        if type(pt) is not type(T.u32) else self.kx_generate(pt, depth + 1))
        call = Call(self._base(h.ret), fn=h.name, args=args,
                    result_ty=self._base(h.ret))
        if type(self._base(t)) is type(self._base(h.ret)):
            return call
        return Cast(t, to=t, arg=call)

    def _warp_vote(self, t: GType, depth: int) -> Expr:
        pred = self._comparison(T.boolean, depth + 1)
        mask = Lit(T.u32, value=0xFFFFFFFF)
        kind = self.ctx.rnd_choice(["ballot_sync", "all_sync", "any_sync"])
        vote = WarpVote(T.u32 if kind == "ballot_sync" else T.i32,
                        kind=kind, mask=mask, predicate=pred)
        base = self._base(t)
        if isinstance(base, TBool):
            return RelOp(T.boolean, op=Cmp.NE, lhs=vote,
                         rhs=Lit(T.i32, value=0))
        return Cast(t, to=t, arg=vote)
