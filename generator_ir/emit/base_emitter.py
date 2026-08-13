"""Backend-neutral source emission for the gpuperf IR.

Provides:
  * ``SourceWriter`` -- indentation-aware line buffer
  * ``SAFE_*`` macro prelude (UB-proof div/mod/shift/add/sub/mul)
  * ``BaseEmitter``  -- expression/statement/function rendering shared by the
    CUDA and HIP backends
"""

from __future__ import annotations

import math
from typing import List

from generator_ir.ir.typedecl import (
    GType, MemSpace, Qual, TVoid, TBool, TChar, TSChar, TShort, TUShort,
    TInt, TUInt, TLong, TULong, THalf, TBF16, TFloat, TDouble,
    TVec, TArray, TStruct, TPointer, TQualified,
)
from generator_ir.ir.operators import SafeOp
from generator_ir.ir.nodes import (
    Expr, Lit, Var, ThreadBuiltin, BinOp, RelOp, LogicOp, UnOp, Select,
    Cast, Fma, SafeBin, Subscript, Member, AddrOf, Deref, Call, Atomic,
    WarpShuffle, WarpVote,
)
from generator_ir.ir.statements import (
    Stmt, Assign, CompoundAssign, IncDecStmt, ExprStmt, Declare, Block,
    If, For, While, DoWhile, Switch, Return, Break, Continue,
    Barrier, Fence, UnrollHint, RawCode,
)
from generator_ir.ir.program import Func, StructDecl


# ---------------------------------------------------------------------------
# Line buffer
# ---------------------------------------------------------------------------

class SourceWriter:
    """Indentation-aware source buffer."""

    def __init__(self, kx_indent: str = "    ") -> None:
        self._lines: List[str] = []
        self._depth = 0
        self._indent = kx_indent

    def kx_indent(self) -> None:
        self._depth += 1

    def kx_dedent(self) -> None:
        self._depth = max(0, self._depth - 1)

    def line(self, text: str) -> None:
        self._lines.append(self._indent * self._depth + text)

    def raw(self, text: str) -> None:
        self._lines.append(text)

    def blank(self) -> None:
        self._lines.append("")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"


# ---------------------------------------------------------------------------
# Safe-op macro prelude
# ---------------------------------------------------------------------------

def write_safe_macros(w: SourceWriter) -> None:
    """UB-proof arithmetic macros emitted at the top of every generated file.

    SDIV/SMOD guard a zero divisor and the signed min / -1 overflow; the
    result is 0 in the guarded case (defined, deterministic).
    SSHL/SSHR mask out-of-range shift counts (result 0 when out of range).
    SADD/SSUB/SMUL evaluate through unsigned long long so signed overflow is
    defined wraparound, then cast back to the promoted operand type.
    """
    w.raw("// UB-proof arithmetic (division/modulo guard, shift masking,")
    w.raw("// signed-via-unsigned overflow)")
    w.raw("#define GPU_SDIV(a, b) ((b) != 0 && "
          "!((__typeof__(b))(-1) < (__typeof__(b))(0) && "
          "(b) == (__typeof__(b))(-1) && "
          "(a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) "
          "? (a) / (b) : 0)")
    w.raw("#define GPU_SMOD(a, b) ((b) != 0 && "
          "!((__typeof__(b))(-1) < (__typeof__(b))(0) && "
          "(b) == (__typeof__(b))(-1) && "
          "(a) == (__typeof__(a))(1ULL << (sizeof(a) * 8 - 1))) "
          "? (a) % (b) : 0)")
    w.raw("#define GPU_SSHL(a, b) "
          "((b) >= 0 && (b) < (long long)(sizeof(a) * 8) "
          "? ((unsigned long long)(a) << (b)) : 0ULL)")
    w.raw("#define GPU_SSHR(a, b) "
          "((b) >= 0 && (b) < (long long)(sizeof(a) * 8) "
          "? (a) >> (b) : 0)")
    w.raw("#define GPU_SADD(a, b) "
          "((__typeof__((a) + (b)))((unsigned long long)(a) "
          "+ (unsigned long long)(b)))")
    w.raw("#define GPU_SSUB(a, b) "
          "((__typeof__((a) + (b)))((unsigned long long)(a) "
          "- (unsigned long long)(b)))")
    w.raw("#define GPU_SMUL(a, b) "
          "((__typeof__((a) + (b)))((unsigned long long)(a) "
          "* (unsigned long long)(b)))")
    w.blank()


def write_diverge_macros(w: SourceWriter) -> None:
    """Divergence illusion macros (uniform at runtime, opaque to the compiler)."""
    w.raw("// Divergence-illusion macros")
    w.raw("#define GPU_FAKE_DIVERGE(idx, off) ((idx) - (off))")
    w.raw("#define GPU_GROUP_DIVERGE(idx, val) ((idx) * 0 + (val))")
    w.blank()


_SAFE_MACRO_NAME = {
    SafeOp.SDIV: "GPU_SDIV",
    SafeOp.SMOD: "GPU_SMOD",
    SafeOp.SSHL: "GPU_SSHL",
    SafeOp.SSHR: "GPU_SSHR",
    SafeOp.SADD: "GPU_SADD",
    SafeOp.SSUB: "GPU_SSUB",
    SafeOp.SMUL: "GPU_SMUL",
}


# ---------------------------------------------------------------------------
# Base emitter
# ---------------------------------------------------------------------------

class BaseEmitter:
    """Renders IR to C-family source.  Backend subclasses override type-name
    and API-name hooks."""

    def __init__(self) -> None:
        self.w = SourceWriter()

    # ---- type names -----------------------------------------------------

    def type_name(self, t: GType) -> str:
        if isinstance(t, TQualified):
            return self._qualified_name(t)
        if isinstance(t, TVoid):
            return "void"
        if isinstance(t, TBool):
            return "bool"
        if isinstance(t, TChar):
            return "unsigned char"
        if isinstance(t, TSChar):
            return "signed char"
        if isinstance(t, TShort):
            return "short"
        if isinstance(t, TUShort):
            return "unsigned short"
        if isinstance(t, TInt):
            return "int"
        if isinstance(t, TUInt):
            return "unsigned int"
        if isinstance(t, TLong):
            return "long long"
        if isinstance(t, TULong):
            return "unsigned long long"
        if isinstance(t, TFloat):
            return "float"
        if isinstance(t, TDouble):
            return "double"
        if isinstance(t, THalf):
            return "__half"
        if isinstance(t, TBF16):
            return self.bfloat16_name()
        if isinstance(t, TVec):
            return t.spelling()
        if isinstance(t, TArray):
            return self.type_name(t.scalar_elem())
        if isinstance(t, TPointer):
            return f"{self.type_name(t.base)}*"
        if isinstance(t, TStruct):
            return t.tag
        return t.spelling()

    def bfloat16_name(self) -> str:
        return "__nv_bfloat16"

    def _qualified_name(self, qt: TQualified) -> str:
        parts: List[str] = []
        for q in qt.quals:
            if q is Qual.CONST:
                parts.append("const")
            elif q is Qual.VOLATILE:
                parts.append("volatile")
        parts.append(self.type_name(qt.base))
        return " ".join(parts)

    # ---- expressions ------------------------------------------------------

    def expr(self, e: Expr) -> str:
        if isinstance(e, Lit):
            return self._lit(e)
        if isinstance(e, Var):
            return e.name
        if isinstance(e, ThreadBuiltin):
            return self._kx_thread_builtin(e)
        if isinstance(e, BinOp):
            return f"({self.expr(e.lhs)} {e.op.symbol} {self.expr(e.rhs)})"
        if isinstance(e, RelOp):
            return f"({self.expr(e.lhs)} {e.op.symbol} {self.expr(e.rhs)})"
        if isinstance(e, LogicOp):
            return f"({self.expr(e.lhs)} {e.op.symbol} {self.expr(e.rhs)})"
        if isinstance(e, UnOp):
            return f"({e.op.symbol}({self.expr(e.arg)}))"
        if isinstance(e, Select):
            return (f"({self.expr(e.cond)} ? {self.expr(e.when_true)}"
                    f" : {self.expr(e.when_false)})")
        if isinstance(e, Cast):
            return self._cast(e)
        if isinstance(e, Fma):
            return self._fma(e)
        if isinstance(e, SafeBin):
            return f"{_SAFE_MACRO_NAME[e.op]}({self.expr(e.lhs)}, {self.expr(e.rhs)})"
        if isinstance(e, Subscript):
            return self._subscript(e)
        if isinstance(e, Member):
            return f"{self.expr(e.obj)}.{e.field}"
        if isinstance(e, Call):
            return self._call(e)
        if isinstance(e, Atomic):
            return self._atomic(e)
        if isinstance(e, WarpShuffle):
            return self._warp_shuffle(e)
        if isinstance(e, WarpVote):
            return self._warp_vote(e)
        if isinstance(e, AddrOf):
            return f"(&({self.expr(e.arg)}))"
        if isinstance(e, Deref):
            return f"(*({self.expr(e.arg)}))"
        return f"/* <unknown expr {type(e).__name__}> */"

    def _lit(self, c: Lit) -> str:
        v = c.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            if math.isinf(v):
                return "(-INFINITY)" if v < 0 else "INFINITY"
            if math.isnan(v):
                return "NAN"
            s = repr(v)
            if isinstance(c.ty, TFloat):
                return s + ("f" if ("e" in s or "." in s) else ".0f")
            if isinstance(c.ty, TDouble):
                return s if ("e" in s or "." in s) else s + ".0"
            if isinstance(c.ty, THalf):
                if "." not in s and "e" not in s:
                    s += ".0"
                return f"__float2half_rn({s}f)"
            if isinstance(c.ty, TBF16):
                if "." not in s and "e" not in s:
                    s += ".0"
                return f"{self._float_to_bf16()}({s}f)"
            return s
        t = c.ty
        if isinstance(t, TUInt):
            return f"{v}u"
        if isinstance(t, TULong):
            return f"{v}ull"
        if isinstance(t, TLong):
            return f"{v}ll"
        if isinstance(t, TSChar):
            return f"((signed char){v})"
        if isinstance(t, TChar):
            return f"((unsigned char){v}u)"
        if isinstance(t, TShort):
            return f"((short){v})"
        if isinstance(t, TUShort):
            return f"((unsigned short){v}u)"
        return str(v)

    def _float_to_bf16(self) -> str:
        return "__float2bfloat16_rn"

    def _kx_thread_builtin(self, t: ThreadBuiltin) -> str:
        if t.source == "warpSize":
            return "warpSize"
        return f"{t.source}.{t.axis}"

    def _cast(self, c: Cast) -> str:
        inner = self.expr(c.arg)
        if c.soft:
            return f"({inner})"
        return f"(({self.type_name(c.to)})({inner}))"

    def _fma(self, f: Fma) -> str:
        a = self.expr(f.a)
        b = self.expr(f.b)
        c = self.expr(f.c)
        if isinstance(f.ty, TFloat):
            return f"fmaf({a}, {b}, {c})"
        if isinstance(f.ty, TDouble):
            # __fma_rn: unambiguous device FMA (the bare fma() can resolve to a
            # host-only overload under nvcc without relaxed-constexpr).
            return f"__fma_rn({a}, {b}, {c})"
        return f"(({a}) * ({b}) + ({c}))"

    def _subscript(self, s: Subscript) -> str:
        base = self.expr(s.array)
        idx = self.expr(s.index)
        if s.stride != 1:
            arr_name = getattr(s.array, "name", "")
            if arr_name == "input":
                # Read-only input buffer sized n: wrap the strided index so it
                # stays in bounds.
                return (f"{base}[GPU_SMOD(({idx}) * {s.stride},"
                        f" ((unsigned int)(n)))]")
            return f"{base}[{idx}]"
        return f"{base}[{idx}]"

    def _atomic(self, a: Atomic) -> str:
        addr = self.expr(a.addr)
        val = self.expr(a.operand)
        if isinstance(a.addr, AddrOf):
            return f"atomic{a.what}{a.kx_scope}({addr}, {val})"
        return f"atomic{a.what}{a.kx_scope}(&({addr}), {val})"

    def _call(self, c: Call) -> str:
        return f"{c.fn}({', '.join(self.expr(a) for a in c.args)})"

    def _warp_vote(self, wv: WarpVote) -> str:
        return f"__{wv.kind}({self.expr(wv.mask)}, {self.expr(wv.predicate)})"

    def _warp_shuffle(self, ws: WarpShuffle) -> str:
        mask = self.expr(ws.mask)
        val = self.expr(ws.value)
        delta = self.expr(ws.delta)
        if ws.width != 32:
            return f"__{ws.kind}({mask}, {val}, {delta}, {ws.width})"
        return f"__{ws.kind}({mask}, {val}, {delta})"

    # ---- statements ---------------------------------------------------------

    def stmt(self, s: Stmt) -> None:
        w = self.w
        if isinstance(s, Assign):
            w.line(f"{self.expr(s.target)} = {self.expr(s.value)};")
        elif isinstance(s, CompoundAssign):
            w.line(f"{self.expr(s.target)} {s.op.symbol} {self.expr(s.value)};")
        elif isinstance(s, IncDecStmt):
            sym = s.op.symbol
            w.line(f"{sym}{self.expr(s.target)};" if s.prefix
                   else f"{self.expr(s.target)}{sym};")
        elif isinstance(s, ExprStmt):
            w.line(f"{self.expr(s.expr)};")
        elif isinstance(s, Declare):
            self._declare(s)
        elif isinstance(s, Block):
            self._block(s)
        elif isinstance(s, If):
            self._if(s)
        elif isinstance(s, For):
            self._for(s)
        elif isinstance(s, While):
            w.line(f"while ({self.expr(s.cond)}) {{")
            w.kx_indent()
            self._stmts(s.body)
            w.kx_dedent()
            w.line("}")
        elif isinstance(s, DoWhile):
            w.line("do {")
            w.kx_indent()
            self._stmts(s.body)
            w.kx_dedent()
            w.line(f"}} while ({self.expr(s.cond)});")
        elif isinstance(s, Switch):
            self._switch(s)
        elif isinstance(s, Return):
            w.line(f"return {self.expr(s.value)};" if s.value is not None
                   else "return;")
        elif isinstance(s, Break):
            w.line("break;")
        elif isinstance(s, Continue):
            w.line("continue;")
        elif isinstance(s, Barrier):
            w.line("__syncthreads();")
        elif isinstance(s, Fence):
            if s.kx_scope == "device":
                w.line("__threadfence();")
            else:
                w.line(f"__threadfence_{s.kx_scope}();")
        elif isinstance(s, UnrollHint):
            w.raw("#pragma unroll" if s.factor is None
                  else f"#pragma unroll {s.factor}")
        elif isinstance(s, RawCode):
            for ln in s.text.split("\n"):
                w.line(ln)
        else:
            w.line(f"/* <unknown stmt {type(s).__name__}> */")

    def _stmts(self, block: Block) -> None:
        for s in block.stmts:
            self.stmt(s)

    def _declare(self, d: Declare) -> None:
        parts: List[str] = []
        if d.space is MemSpace.SHARED:
            parts.append("__shared__")
        elif d.space is MemSpace.CONSTANT:
            parts.append("__constant__")
        for q in d.quals:
            if q is Qual.CONST:
                parts.append("const")
            elif q is Qual.VOLATILE:
                parts.append("volatile")
            elif q is Qual.RESTRICT:
                parts.append("__restrict__")
            elif q is Qual.DEVICE:
                parts.append("__device__")
            elif q is Qual.MANAGED:
                parts.append("__managed__")
        if isinstance(d.gtype, TArray):
            inner: GType = d.gtype
            dims: List[int] = []
            while isinstance(inner, TArray):
                dims.append(inner.count)
                inner = inner.elem
            parts.append(self.type_name(inner))
            name = d.name + "".join(f"[{n}]" for n in dims)
        else:
            parts.append(self.type_name(d.gtype))
            name = d.name
        text = " ".join(parts) + " " + name
        if d.init is not None:
            text += f" = {self.expr(d.init)}"
        self.w.line(text + ";")

    def _block(self, b: Block) -> None:
        self.w.line("{")
        self.w.kx_indent()
        self._stmts(b)
        self.w.kx_dedent()
        self.w.line("}")

    def _if(self, s: If) -> None:
        self.w.line(f"if ({self.expr(s.cond)}) {{")
        self.w.kx_indent()
        self._stmts(s.then)
        self.w.kx_dedent()
        if s.otherwise is not None:
            self.w.line("} else {")
            self.w.kx_indent()
            self._stmts(s.otherwise)
            self.w.kx_dedent()
        self.w.line("}")

    def _for(self, s: For) -> None:
        self.w.line(
            f"for (int {s.var} = {self.expr(s.start)}; {s.var} < {self.expr(s.end)};"
            f" {s.var} += {self.expr(s.step)}) {{")
        self.w.kx_indent()
        self._stmts(s.body)
        self.w.kx_dedent()
        self.w.line("}")

    def _switch(self, s: Switch) -> None:
        self.w.line(f"switch ({self.expr(s.expr)}) {{")
        self.w.kx_indent()
        for c in s.cases:
            if c.kx_is_default():
                self.w.line("default: {")
            else:
                self.w.line(f"case {self.expr(c.value)}: {{")
            self.w.kx_indent()
            self._stmts(c.body)
            if c.has_break:
                self.w.line("break;")
            self.w.kx_dedent()
            self.w.line("}")
        self.w.kx_dedent()
        self.w.line("}")

    # ---- declarations ------------------------------------------------------

    def function(self, f: Func) -> None:
        sig = self.signature(f)
        if not f.kx_has_body():
            self.w.line(f"{sig};")
            return
        self.w.line(f"{sig} {{")
        self.w.kx_indent()
        self._stmts(f.body)
        self.w.kx_dedent()
        self.w.line("}")

    def signature(self, f: Func) -> str:
        parts: List[str] = []
        if f.inline_hint:
            parts.append(f.inline_hint)
        if f.kx_is_kernel():
            parts.append("__global__")
        elif f.kx_is_device():
            parts.append("__device__")
        if f.launch_bounds and f.kx_is_kernel():
            parts.append(f"__launch_bounds__({', '.join(str(v) for v in f.launch_bounds)})")
        if f.maxnreg is not None and f.kx_is_kernel():
            parts.append(f"__maxnreg__({f.maxnreg})")
        if f.cluster_dims is not None and f.kx_is_kernel():
            parts.append(f"__cluster_dims__({', '.join(str(v) for v in f.cluster_dims)})")
        parts.append(self.type_name(f.ret))
        params = ", ".join(self._param(p) for p in f.params)
        return " ".join(parts) + f" {f.name}({params})"

    def _param(self, p: Declare) -> str:
        pre: List[str] = []
        post: List[str] = []
        for q in p.quals:
            if q is Qual.CONST:
                pre.append("const")
            elif q is Qual.RESTRICT:
                post.append("__restrict__")
            elif q is Qual.GRID_CONSTANT:
                pre.append("__grid_constant__")
            elif q is Qual.VOLATILE:
                pre.append("volatile")
        if isinstance(p.gtype, TArray):
            inner: GType = p.gtype
            dims: List[int] = []
            while isinstance(inner, TArray):
                dims.append(inner.count)
                inner = inner.elem
            head = " ".join(pre + [self.type_name(inner)])
            return f"{head} {p.name}" + "".join(f"[{n}]" for n in dims)
        head = " ".join(pre + [self.type_name(p.gtype)])
        tail = (" " + " ".join(post)) if post else ""
        return f"{head}{tail} {p.name}"

    def struct(self, sd: StructDecl) -> None:
        self.w.line(f"struct {sd.struct.tag} {{")
        self.w.kx_indent()
        for fname, ftype in sd.struct.fields:
            if isinstance(ftype, TArray):
                self.w.line(f"{self.type_name(ftype.elem)} {fname}[{ftype.count}];")
            else:
                self.w.line(f"{self.type_name(ftype)} {fname};")
        self.w.kx_dedent()
        self.w.line("};")
