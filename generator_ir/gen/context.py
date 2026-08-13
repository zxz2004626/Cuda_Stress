"""Generation context: RNG, scopes, safety flags, and the weaving builder API."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from generator_ir.gen.scope_tracker import VarScope
from generator_ir.ir.typedecl import (
    GType, T, TBool, TChar, TSChar, TShort, TUShort, TInt, TUInt, TLong,
    TULong, TFloat, TDouble, THalf, TBF16, TVec, TArray, TPointer,
    TQualified, Qual, MemSpace, Uniformity,
)
from generator_ir.ir.nodes import Expr, Var, Lit, Cast
from generator_ir.ir.statements import Declare
from generator_ir.gen.options import GenConfig


@dataclass(frozen=True)
class CtxSnapshot:
    """Generation-context snapshot stamped onto emitted statements."""
    uniform: bool = True
    loop_depth: int = 0
    if_depth: int = 0
    in_helper: bool = False
    kx_is_2d: bool = False
    inside_breakable_loop: bool = False


class GenContext:
    """Mutable state threaded through every generator."""

    # Names that must never be assigned to by a generated statement.
    RESERVED_NAMES = frozenset({
        "tid", "gid", "n", "output", "input", "_fv", "_rep", "_ci",
        "x", "y", "idx", "output_fp_acc",
    })

    # Scalar types accepted as "integer-ish" by the weaving pool.
    POOL_TYPES = (
        TUInt, TInt, TULong, TLong, TUShort, TShort, TChar, TSChar, TBool,
    )

    def __init__(self, config: GenConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

        p_float = getattr(config, "p_float_types", 0.0)
        self.use_float_types = p_float > 0.0 and self.rng.random() < p_float

        from generator_ir.gen.weights import kx_schedule_for_seed
        self.weights = kx_schedule_for_seed(config.seed)

        self._scope = VarScope()

        self.expr_depth = 0
        self.loop_depth = 0
        self.if_depth = 0
        self.stmt_count = 0

        # Product of enclosing loop trip counts (runaway guard).
        self.loop_trip_product = 1
        self.max_loop_trip_product = 1024

        self._function_scope_names: Set[str] = set()

        # Safety flags (barrier legality etc.)
        self.inside_breakable_loop = False
        self.in_nested_control_flow = False
        self.in_device_helper = False

        self._name_counters: Dict[str, int] = {}
        self._varying_vars: Set[str] = set()
        self.features_used: Set[str] = set()
        self.kx_is_2d = False

        # Weaving builder state
        self._blocks: List[List] = []
        self._fv_var: Optional[Var] = Var(T.u32, name="_fv")
        self._fv_declared = False
        self._fp_var: Optional[Var] = Var(T.u32, name="output_fp_acc")
        self._fp_declared = False
        self._used_fp = False
        self._weave_schedule = None
        self._block_is_1d = True
        self._weave_mode = False
        self._build_depth = 0
        self._max_build_depth = 2
        self._emit_count = 0
        self._emit_cap = 10 ** 9
        self.fired_blocks: "Counter[str]" = Counter()

        # Feature plumbing flags set by statement generators
        self._needs_dynamic_shared = False
        self._needs_coop_groups = False
        self._needs_coop_groups_memcpy_async = False
        self._needs_coop_groups_reduce = False
        self._needs_wmma = False
        self._needs_constant_mem = False
        self._needs_cuda_barrier = False
        self._device_arrays: List[dict] = []
        self._device_helpers: List = []
        self._visible_names: Optional[Set[str]] = None
        self._preferred_fragment: Optional[str] = None
        self._in_template_fragment = False
        self._fragment_registry = None

    # -- kx_scope ------------------------------------------------------------

    @property
    def kx_scope(self) -> VarScope:
        return self._scope

    def kx_push_scope(self) -> None:
        self._scope = self._scope.push()

    def kx_pop_scope(self) -> None:
        self._scope = self._scope.pop()

    def kx_declare(self, var: Var, divergent: bool = False) -> None:
        self._scope.kx_declare(var)
        if divergent:
            self._varying_vars.add(var.name)
        if self._scope._parent is None:
            self._function_scope_names.add(var.name)

    def kx_function_scope_variables(self) -> List[Var]:
        return [v for v in self._scope.kx_all_variables()
                if v.name in self._function_scope_names]

    # -- variable sampling ----------------------------------------------------

    def kx_sample_or_create(self, gtype: GType, for_write: bool = False,
                         init: Optional[Expr] = None):
        """Pick an in-kx_scope var of *gtype*, or kx_declare one if none matches.

        Returns ``(var, decl_or_None)``; the caller emits the decl.
        """
        candidates = self._scope.kx_variables_of_type(
            gtype, only_names=self._visible_names)
        if for_write:
            candidates = [v for v in candidates
                          if not v.name.startswith("i_")
                          and v.name not in self.RESERVED_NAMES]
        if candidates:
            return self.rnd_choice(candidates), None
        name = self.kx_fresh_name("v")
        var = Var(gtype, name=name)
        self.kx_declare(var)
        decl = Declare(name=name, gtype=gtype,
                       init=init if init is not None else Lit(gtype, value=0))
        return var, decl

    # -- naming / divergence ------------------------------------------------------

    def kx_fresh_name(self, prefix: str = "v") -> str:
        idx = self._name_counters.get(prefix, 0)
        self._name_counters[prefix] = idx + 1
        return f"{prefix}_{idx}"

    def is_varying(self, name: str) -> bool:
        return name in self._varying_vars

    def mark_varying(self, name: str) -> None:
        self._varying_vars.add(name)

    # -- random helpers --------------------------------------------------------------

    def rnd_int(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def rnd_choice(self, seq):
        return self.rng.choice(seq)

    def rnd_float(self) -> float:
        return self.rng.random()

    def rnd_bool(self, prob: float = 0.5) -> bool:
        return self.rng.random() < prob

    def rnd_weighted(self, choices_weights: List[Tuple]):
        choices = [c for c, _ in choices_weights]
        weights = [w for _, w in choices_weights]
        return self.rng.choices(choices, weights=weights, k=1)[0]

    # -- backend / arch ---------------------------------------------------------------

    def is_hip(self) -> bool:
        return getattr(self.config, "backend", "hip") == "hip"

    def kx_arch_number(self) -> int:
        arch = getattr(self.config, "arch", "")
        if arch.startswith("gfx"):
            digits = arch[3:]
            return int(digits) if digits.isdigit() else 0
        if arch.startswith("sm_"):
            digits = arch[3:]
            return int(digits) if digits.isdigit() else 0
        m = re.search(r"(\d+)", arch)
        return int(m.group(1)) if m else 0

    # -- type pools ----------------------------------------------------------------------

    def _unique(self, types: List[GType]) -> List[GType]:
        seen: Set[Tuple[type, str]] = set()
        out: List[GType] = []
        for t in types:
            key = (type(t), t.spelling())
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    def _type_ok(self, t: GType) -> bool:
        if isinstance(t, TVec):
            return (t.elem.kx_min_arch <= self.kx_arch_number()
                    and not isinstance(t.elem, TBF16)
                    and self._type_ok(t.elem))
        kx_min_arch = getattr(t, "kx_min_arch", 0)
        if kx_min_arch and self.kx_arch_number() and kx_min_arch > self.kx_arch_number():
            return False
        return True

    def _parse_type_name(self, name: str) -> Optional[GType]:
        raw = name.strip().lower().replace(" ", "").replace("_", "")
        aliases = {
            "bool": T.boolean, "int": T.i32, "uint": T.u32,
            "unsignedint": T.u32, "char": T.uchar, "uchar": T.uchar,
            "unsignedchar": T.uchar, "signedchar": T.schar, "schar": T.schar,
            "short": T.short, "ushort": T.ushort, "unsignedshort": T.ushort,
            "longlong": T.i64, "unsignedlonglong": T.u64, "ulonglong": T.u64,
            "float": T.f32, "double": T.f64,
            "half": T.f16, "__half": T.f16,
            "bfloat16": T.bf16, "bf16": T.bf16,
        }
        if raw in aliases:
            return aliases[raw]
        m = re.fullmatch(r"([a-z]+?)([234])", raw)
        if not m:
            return None
        base = aliases.get(m.group(1))
        if base is None or isinstance(base, TBF16):
            return None
        return TVec(base, int(m.group(2)))

    def _allowed_types(self) -> List[GType]:
        configured = getattr(self.config, "allowed_types", None)
        if configured is None:
            return []
        out: List[GType] = []
        for name in configured:
            t = self._parse_type_name(str(name))
            if t is not None and self._type_ok(t):
                out.append(t)
        return self._unique(out)

    @staticmethod
    def _scalar_only(types: List[GType]) -> List[GType]:
        return [t for t in types
                if not isinstance(t, (TVec, TArray, TPointer))]

    def _diverse_pool(self) -> List[GType]:
        return self._unique([
            T.schar, T.uchar, T.short, T.ushort, T.i32, T.u32,
            T.i64, T.u64, T.boolean,
        ])

    def kx_computation_types(self) -> List[GType]:
        """Types usable in computation (floats join only via explicit config)."""
        allowed = self._allowed_types()
        if allowed:
            scalar = self._scalar_only(allowed)
            return scalar if scalar else [T.u32, T.i32]
        if self.config.use_full_int_widths:
            return self._diverse_pool()
        return [T.u32, T.i32]

    def declaration_types(self) -> List[GType]:
        allowed = self._allowed_types()
        if allowed:
            scalar = self._scalar_only(allowed)
            if scalar:
                return scalar
            return [T.u32, T.i32]
        if self.config.use_full_int_widths:
            return self._diverse_pool()
        return [T.u32, T.i32]

    def kx_primary_type(self) -> GType:
        return T.u32

    def kx_safe_arith_ops(self):
        from generator_ir.ir.operators import Arith
        return [Arith.ADD, Arith.SUB, Arith.MUL,
                Arith.BAND, Arith.BOR, Arith.BXOR]

    # ==========================================================================
    # Weaving builder API (two-tier codegen)
    # ==========================================================================

    @property
    def snapshot(self) -> CtxSnapshot:
        uniform = (self.if_depth == 0
                   and not self.inside_breakable_loop
                   and not self.in_nested_control_flow)
        return CtxSnapshot(
            uniform=uniform,
            loop_depth=self.loop_depth,
            if_depth=self.if_depth,
            in_helper=self.in_device_helper,
            kx_is_2d=self.kx_is_2d,
            inside_breakable_loop=self.inside_breakable_loop,
        )

    def emit(self, stmt) -> None:
        """Append *stmt* to the current weaving block, stamping its context."""
        if not self._blocks:
            self._blocks.append([])
        if getattr(stmt, "emit_ctx", None) is None:
            stmt.emit_ctx = self.snapshot
        self._blocks[-1].append(stmt)
        self._emit_count += 1

    def kx_open_block(self) -> None:
        self._blocks.append([])
        self.kx_push_scope()

    def kx_close_block(self) -> List:
        assert self._blocks, "kx_close_block with no open block"
        stmts = self._blocks.pop()
        self.kx_pop_scope()
        return stmts

    def kx_current_block(self) -> List:
        if not self._blocks:
            self._blocks.append([])
        return self._blocks[-1]

    def _live_pool_vars(self, for_write: bool) -> List[Var]:
        out: List[Var] = []
        for v in self._scope.kx_all_variables():
            if self._visible_names is not None and v.name not in self._visible_names:
                continue
            vt = v.ty
            if isinstance(vt, TQualified):
                vt = vt.unqualified()
            if isinstance(vt, (TPointer, TArray)):
                continue
            if not isinstance(vt, self.POOL_TYPES):
                continue
            if for_write and (v.name.startswith("i_")
                              or v.name in self.RESERVED_NAMES):
                continue
            out.append(v)
        return out

    def kx_random_var(self, of_type: GType, *, loose: bool = True,
                   recency: float = 0.5, for_write: bool = False):
        """Pick a live variable coercible to *of_type* (loose = any int pool
        var, cast as needed).  for_write=True returns an assignable Var."""
        candidates = self._live_pool_vars(for_write)
        if not loose:
            candidates = [v for v in candidates
                          if type(self._kx_unqual(v.ty)) is type(of_type)]
        if not candidates:
            if for_write:
                return self.fresh(of_type, init=Lit(of_type, value=0))
            return Lit(of_type, value=self.rnd_int(0, 7))

        var = self._recency_pick(candidates, recency)
        if for_write:
            return Var(var.ty, name=var.name)
        ref = Var(var.ty, name=var.name)
        if type(self._kx_unqual(var.ty)) is type(of_type):
            return ref
        return Cast(of_type, to=of_type, arg=ref)

    @staticmethod
    def _kx_unqual(t: GType) -> GType:
        return t.unqualified() if isinstance(t, TQualified) else t

    def _recency_pick(self, candidates: List[Var], recency: float):
        if recency <= 0.0 or len(candidates) == 1:
            return self.rnd_choice(candidates)
        n = len(candidates)
        weights = [1.0 + recency * (i / max(1, n - 1)) * n for i in range(n)]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def fresh(self, of_type: GType, init: Optional[Expr] = None,
              *, divergent: bool = False) -> Var:
        """Declare AND emit a new pool variable; return the Var."""
        name = self.kx_fresh_name("cv")
        var = Var(of_type, name=name)
        self.kx_declare(var, divergent=divergent)
        self.emit(Declare(name=name, gtype=of_type,
                          init=init if init is not None else Lit(of_type, value=0)))
        return var

    # -- bootstrap ------------------------------------------------------------

    def bootstrap(self) -> None:
        """Seed the weaving pool; marks weave mode + observable fold on."""
        self._weave_mode = True
        if not getattr(self.config, "observable_fold", False):
            self.config.observable_fold = True
        self._max_build_depth = max(1, self.kx_recursion_depth_for_complexity())
        self._emit_cap = max(80, self.kx_budget_for_complexity() * 6)
        self._kx_ensure_fv()

        tid = self._scope.lookup("tid")
        if tid is not None:
            self.fresh(T.u32, init=Cast(T.u32, to=T.u32,
                                        arg=Var(T.u32, name="tid")),
                       divergent=True)
        self.fresh(T.u32, init=Lit(T.u32, value=self.rnd_int(1, 0xFFFF)))
        self.fresh(T.u64, init=Lit(T.u64, value=self.rnd_int(1, 0xFFFFFF)))

    # -- observed-value emission --------------------------------------------------

    def build_observed_int(self, target_type: Optional[GType] = None, depth: int = 0):
        """An integer expression guaranteed safe to fold into the checksum."""
        from generator_ir.gen.expr_builder import ExprGen
        if target_type is None:
            target_type = T.u32
        if not getattr(self.config, "observable_fold", False):
            self.config.observable_fold = True
        expr = ExprGen(self).kx_generate(target_type, depth)
        et = self._kx_unqual(getattr(expr, "ty", target_type))
        if not isinstance(et, self.POOL_TYPES):
            return Cast(target_type, to=target_type, arg=expr)
        return expr

    def _render_root_decl(self, stmt) -> None:
        if getattr(stmt, "emit_ctx", None) is None:
            stmt.emit_ctx = self.snapshot
        if not self._blocks:
            self._blocks.append([])
        self._blocks[0].insert(0, stmt)

    def _kx_ensure_fv(self) -> Var:
        if not self._fv_declared:
            self._render_root_decl(Declare(name="_fv", gtype=T.u32,
                                         init=Lit(T.u32, value=0)))
            self._fv_declared = True
        return self._fv_var

    def observe(self, value: Expr) -> None:
        """``_fv = _fv * K ^ (unsigned)value;``"""
        from generator_ir.ir.nodes import BinOp
        from generator_ir.ir.operators import Arith
        from generator_ir.ir.statements import Assign
        self._kx_ensure_fv()
        val_u = Cast(T.u32, to=T.u32, arg=value)
        mixed = BinOp(T.u32, op=Arith.BXOR,
                      lhs=BinOp(T.u32, op=Arith.MUL,
                                lhs=Var(T.u32, name="_fv"),
                                rhs=Lit(T.u32, value=0x9E3779B1)),
                      rhs=val_u)
        self.emit(Assign(Var(T.u32, name="_fv"), mixed))

    def kx_observe_fp(self, value: Expr) -> None:
        """Fold an FP value into the off-checksum ``output_fp_acc``."""
        from generator_ir.ir.nodes import BinOp, Call
        from generator_ir.ir.operators import Arith
        from generator_ir.ir.statements import Assign
        if not self._fp_declared:
            self._render_root_decl(Declare(name="output_fp_acc", gtype=T.u32,
                                         init=Lit(T.u32, value=0)))
            self._fp_declared = True
        self._used_fp = True

        vt = self._kx_unqual(value.ty)
        if isinstance(vt, TDouble):
            as_u = Cast(T.u32, to=T.u32,
                        arg=Call(T.u64, fn="__double_as_longlong",
                                 args=[value], result_ty=T.u64))
        elif isinstance(vt, TFloat):
            as_u = Call(T.u32, fn="__float_as_uint", args=[value], result_ty=T.u32)
        else:
            as_u = Cast(T.u32, to=T.u32, arg=value)
        self.emit(Assign(Var(T.u32, name="output_fp_acc"),
                         BinOp(T.u32, op=Arith.BXOR,
                               lhs=Var(T.u32, name="output_fp_acc"), rhs=as_u)))

    def used_fp(self) -> bool:
        return self._used_fp

    def render_observe_write(self, output_var: Var, index_expr: Optional[Expr] = None,
                           xor: bool = False) -> None:
        """Emit the final observed output write into the current block.

        1D: output[index] = _fv;  2D: output[idx] ^= _fv.  In weave mode an
        extra read-mix-write keeps the fold observably alive.
        """
        from generator_ir.ir.nodes import BinOp, Subscript
        from generator_ir.ir.operators import Arith
        from generator_ir.ir.statements import Assign
        self._kx_ensure_fv()
        idx = index_expr if index_expr is not None else Var(T.u32, name="tid")

        def slot() -> Subscript:
            return Subscript(T.u32, array=output_var, index=idx)

        if xor:
            self.emit(Assign(slot(), BinOp(T.u32, op=Arith.BXOR,
                                           lhs=slot(), rhs=Var(T.u32, name="_fv"))))
        else:
            self.emit(Assign(slot(), Var(T.u32, name="_fv")))

        if not getattr(self.config, "codegen", False):
            return

        obs_name = self.kx_fresh_name("obs")
        obs = Var(T.u32, name=obs_name)
        self.emit(Declare(name=obs_name, gtype=T.u32, init=slot()))
        mix1 = BinOp(T.u32, op=Arith.BXOR,
                     lhs=BinOp(T.u32, op=Arith.MUL, lhs=obs,
                               rhs=Lit(T.u32, value=0x9E3779B1)),
                     rhs=BinOp(T.u32, op=Arith.ADD,
                               lhs=Var(T.u32, name="_fv"),
                               rhs=Var(T.u32, name="tid")))
        self.emit(Assign(obs, mix1))
        self.emit(Assign(slot(), obs))

        if getattr(self.config, "complexity", "medium") in ("high", "stress"):
            mix2 = BinOp(T.u32, op=Arith.BXOR,
                         lhs=slot(),
                         rhs=BinOp(T.u32, op=Arith.ADD,
                                   lhs=BinOp(T.u32, op=Arith.SHR,
                                             lhs=Var(T.u32, name="_fv"),
                                             rhs=Lit(T.u32, value=11)),
                                   rhs=Lit(T.u32, value=0x85EBCA6B)))
            self.emit(Assign(obs, mix2))
            self.emit(Assign(slot(), obs))

    # -- budgets / weaving loop -----------------------------------------------------

    def kx_budget_for_complexity(self) -> int:
        base = {"low": 36, "medium": 95, "high": 140, "stress": 340}.get(
            getattr(self.config, "complexity", "medium"), 95)
        jitter = self.rnd_int(-base // 5, base // 5)
        return max(8, base + jitter)

    def kx_recursion_depth_for_complexity(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "stress": 3}.get(
            getattr(self.config, "complexity", "medium"), 2)

    @property
    def weave_schedule(self) -> "object":
        if self._weave_schedule is None:
            from generator_ir.gen.weights import kx_schedule_for_seed
            self._weave_schedule = kx_schedule_for_seed(self.config.seed, mode="weave")
        return self._weave_schedule

    def weave(self, budget: int) -> None:
        """Repeatedly run a runnable weaving block until budget is spent."""
        from generator_ir.gen.weave_handlers import pick_block
        guard = 0
        while (budget > 0 and self._emit_count < self._emit_cap
               and guard < budget * 8 + 64):
            guard += 1
            block = pick_block(self)
            if block is None:
                break
            self.fired_blocks[block.name] += 1
            block.run(self)
            budget -= max(1, int(getattr(block, "cost", 1)))

    def weave_nested(self, budget: int) -> None:
        """Weave into a nested control-flow body with a hard recursion bound."""
        if (self._build_depth >= self._max_build_depth
                or self._emit_count >= self._emit_cap):
            return
        self._build_depth += 1
        try:
            self.weave(budget)
        finally:
            self._build_depth -= 1
