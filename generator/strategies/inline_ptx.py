"""inline_ptx strategy.

Generates CUDA kernels whose core computation is a *randomly generated inline PTX
block* (asm volatile).  The block is assembled from a small, empirically verified PTX
instruction set (every op was validated against its Python mirror on a real GPU), with:

  * many simultaneously-live 32/64-bit registers (register pressure / spills),
  * long dependency chains plus independent chains (scheduling pressure),
  * predicate registers (setp / and.or.not.pred / selp) live across the block,
  * rare backend-relevant ops (prmt, shf.r, bfe, bfi, lop3, sad, mul.hi, popc, clz, brev),
  * runtime inputs and immediate constants feeding constant banks.

All semantics are unsigned modular; every shift count is masked to a legal range first,
so there is no undefined behaviour.  The CPU oracle simulates the exact same instruction
sequence, so GPU-vs-oracle equality is a true correctness check of the compiled block.

Register naming: nvcc inline-asm substitutes %N operands, and the compiler itself owns
%r/%rd/%p names -- so internal PTX registers use the %%xr/%%xd/%%xp escape (literal %).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import GeneratedCase, Strategy
from .helpers import MASK32, MASK64, make_input_words, pack_words, unpack_words

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Verified PTX instruction table.
# Each op knows how to emit itself and how to evaluate itself.
# ---------------------------------------------------------------------------
def _imm32(x: int) -> str:
    return f"{x & M32}"

def _imm64(x: int) -> str:
    return f"0x{(x & M64):016x}"


def _op(name: str, dst: str, srcs: List[str], imm: Optional[str],
        emit: Callable[[List[str], int], str],
        mirror: Callable[[List[int], int], int]) -> Dict[str, Any]:
    return {"name": name, "dst": dst, "srcs": srcs, "imm": imm,
            "emit": emit, "mirror": mirror}


def _build_op_table() -> List[Dict[str, Any]]:
    O = []
    # ---- b32 binary (reg, reg) -------------------------------------------------
    for name in ("add.u32", "sub.u32", "mul.lo.u32", "and.b32", "or.b32", "xor.b32",
                 "min.u32", "max.u32"):
        O.append(_op(name, "r32", ["r32", "r32"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=name: _bin32(v, n)))
    O.append(_op("mul.hi.u32", "r32", ["r32", "r32"], None,
                 lambda d, s: f"mul.hi.u32 {d}, {s[0]}, {s[1]};",
                 lambda v: ((v[0] * v[1]) >> 32) & M32))
    # ---- b32 unary --------------------------------------------------------------
    O.append(_op("popc.b32", "r32", ["r32"], None,
                 lambda d, s: f"popc.b32 {d}, {s[0]};",
                 lambda v: bin(v[0]).count("1")))
    O.append(_op("clz.b32", "r32", ["r32"], None,
                 lambda d, s: f"clz.b32 {d}, {s[0]};",
                 lambda v: 32 - v[0].bit_length() if v[0] else 32))
    O.append(_op("brev.b32", "r32", ["r32"], None,
                 lambda d, s: f"brev.b32 {d}, {s[0]};",
                 lambda v: int(f"{v[0] & M32:032b}"[::-1], 2)))
    # ---- b32 shifts: register count MUST be pre-masked by the generator ---------
    O.append(_op("shl.b32", "r32", ["r32", "r32"], "shift32",
                 lambda d, s: f"shl.b32 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] << (v[1] & 31)) & M32))
    O.append(_op("shr.u32", "r32", ["r32", "r32"], "shift32",
                 lambda d, s: f"shr.u32 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] >> (v[1] & 31)) & M32))
    O.append(_op("shl.b32_i", "r32", ["r32"], "imm5",
                 lambda d, s, i: f"shl.b32 {d}, {s[0]}, {i & 31};",
                 lambda v, i: (v[0] << (i & 31)) & M32))
    O.append(_op("shr.u32_i", "r32", ["r32"], "imm5",
                 lambda d, s, i: f"shr.u32 {d}, {s[0]}, {i & 31};",
                 lambda v, i: (v[0] >> (i & 31)) & M32))
    # ---- b32 rare / special ------------------------------------------------------
    O.append(_op("bfe.u32", "r32", ["r32"], "bfe",
                 lambda d, s, i: f"bfe.u32 {d}, {s[0]}, {i[0]}, {i[1]};",
                 lambda v, i: (v[0] >> i[0]) & ((1 << i[1]) - 1)))
    O.append(_op("bfi.b32", "r32", ["r32", "r32"], "bfi",
                 lambda d, s, i: f"bfi.b32 {d}, {s[0]}, {s[1]}, {i[0]}, {i[1]};",
                 lambda v, i: _bfi(v[0], v[1], i[0], i[1])))
    O.append(_op("prmt.b32", "r32", ["r32", "r32"], "prmt",
                 lambda d, s, i: f"prmt.b32 {d}, {s[0]}, {s[1]}, 0x{i:04x};",
                 lambda v, i: _prmt(v[0], v[1], i)))
    O.append(_op("lop3.b32", "r32", ["r32", "r32", "r32"], "lop3",
                 lambda d, s, i: f"lop3.b32 {d}, {s[0]}, {s[1]}, {s[2]}, 0x{i:02x};",
                 lambda v, i: _lop3(v[0], v[1], v[2], i)))
    O.append(_op("sad.u32", "r32", ["r32", "r32", "r32"], None,
                 lambda d, s: f"sad.u32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: (v[2] + (v[0] - v[1] if v[0] >= v[1] else v[1] - v[0])) & M32))
    O.append(_op("shf.r.wrap.b32", "r32", ["r32", "r32"], "imm5",
                 lambda d, s, i: f"shf.r.wrap.b32 {d}, {s[0]}, {s[1]}, {i & 31};",
                 lambda v, i: ((v[0] | (v[1] << 32)) >> (i & 31)) & M32))
    O.append(_op("selp.b32", "r32", ["r32", "r32", "pred"], None,
                 lambda d, s: f"selp.b32 {d}, {s[0]}, {s[1]}, {s[2]};",
                 lambda v: v[0] if v[2] else v[1]))
    # ---- b64 binary ---------------------------------------------------------------
    for name in ("add.u64", "sub.u64", "mul.lo.u64", "and.b64", "or.b64", "xor.b64",
                 "min.u64", "max.u64"):
        O.append(_op(name, "r64", ["r64", "r64"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, n=name: _bin64(v, n)))
    O.append(_op("mul.hi.u64", "r64", ["r64", "r64"], None,
                 lambda d, s: f"mul.hi.u64 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] * v[1]) >> 64))
    O.append(_op("shl.b64", "r64", ["r64", "r64"], "shift64",
                 lambda d, s: f"shl.b64 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] << (v[1] & 63)) & M64))
    O.append(_op("shr.u64", "r64", ["r64", "r64"], "shift64",
                 lambda d, s: f"shr.u64 {d}, {s[0]}, {s[1]};",
                 lambda v: (v[0] >> (v[1] & 63)) & M64))
    O.append(_op("shl.b64_i", "r64", ["r64"], "imm6",
                 lambda d, s, i: f"shl.b64 {d}, {s[0]}, {i & 63};",
                 lambda v, i: (v[0] << (i & 63)) & M64))
    O.append(_op("shr.u64_i", "r64", ["r64"], "imm6",
                 lambda d, s, i: f"shr.u64 {d}, {s[0]}, {i & 63};",
                 lambda v, i: (v[0] >> (i & 63)) & M64))
    # ---- b64 unary / cvt ----------------------------------------------------------
    O.append(_op("popc.b64", "r32", ["r64"], None,
                 lambda d, s: f"popc.b64 {d}, {s[0]};",
                 lambda v: bin(v[0]).count("1")))
    O.append(_op("clz.b64", "r32", ["r64"], None,
                 lambda d, s: f"clz.b64 {d}, {s[0]};",
                 lambda v: 64 - v[0].bit_length() if v[0] else 64))
    O.append(_op("brev.b64", "r64", ["r64"], None,
                 lambda d, s: f"brev.b64 {d}, {s[0]};",
                 lambda v: int(f"{v[0] & M64:064b}"[::-1], 2)))
    O.append(_op("cvt.u64.u32", "r64", ["r32"], None,
                 lambda d, s: f"cvt.u64.u32 {d}, {s[0]};",
                 lambda v: v[0] & M64))
    O.append(_op("cvt.u32.u64", "r32", ["r64"], None,
                 lambda d, s: f"cvt.u32.u64 {d}, {s[0]};",
                 lambda v: v[0] & M32))
    # ---- predicates ---------------------------------------------------------------
    for name, cmp in (("setp.lt.u32", "lt"), ("setp.gt.u32", "gt"), ("setp.eq.u32", "eq"),
                      ("setp.ne.u32", "ne"), ("setp.ge.u32", "ge")):
        O.append(_op(name, "pred", ["r32", "r32"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, c=cmp: _cmp(v[0], v[1], c)))
    for name, cmp in (("setp.lt.u64", "lt"), ("setp.gt.u64", "gt"), ("setp.eq.u64", "eq")):
        O.append(_op(name, "pred", ["r64", "r64"], None,
                     lambda d, s, n=name: f"{n} {d}, {s[0]}, {s[1]};",
                     lambda v, c=cmp: _cmp(v[0], v[1], c)))
    O.append(_op("and.pred", "pred", ["pred", "pred"], None,
                 lambda d, s: f"and.pred {d}, {s[0]}, {s[1]};",
                 lambda v: 1 if (v[0] and v[1]) else 0))
    O.append(_op("or.pred", "pred", ["pred", "pred"], None,
                 lambda d, s: f"or.pred {d}, {s[0]}, {s[1]};",
                 lambda v: 1 if (v[0] or v[1]) else 0))
    O.append(_op("not.pred", "pred", ["pred"], None,
                 lambda d, s: f"not.pred {d}, {s[0]};",
                 lambda v: 0 if v[0] else 1))
    # ---- mov (reg / immediate) ------------------------------------------------------
    O.append(_op("mov.b32", "r32", ["r32"], None,
                 lambda d, s: f"mov.b32 {d}, {s[0]};",
                 lambda v: v[0] & M32))
    O.append(_op("mov.b32_i", "r32", [], "u32",
                 lambda d, s, i: f"mov.b32 {d}, {i & M32};",
                 lambda v, i: i & M32))
    O.append(_op("mov.u64", "r64", ["r64"], None,
                 lambda d, s: f"mov.u64 {d}, {s[0]};",
                 lambda v: v[0] & M64))
    O.append(_op("mov.u64_i", "r64", [], "u64",
                 lambda d, s, i: f"mov.u64 {d}, {_imm64(i)};",
                 lambda v, i: i & M64))
    return O


OPS = _build_op_table()
_OPS_BY_DST = {
    "r32": [o for o in OPS if o["dst"] == "r32"],
    "r64": [o for o in OPS if o["dst"] == "r64"],
    "pred": [o for o in OPS if o["dst"] == "pred"],
}


def _bin32(v, name):
    a, b = v[0] & M32, v[1] & M32
    return {"add.u32": a + b, "sub.u32": a - b, "mul.lo.u32": a * b,
            "and.b32": a & b, "or.b32": a | b, "xor.b32": a ^ b,
            "min.u32": min(a, b), "max.u32": max(a, b)}[name] & M32


def _bin64(v, name):
    a, b = v[0] & M64, v[1] & M64
    return {"add.u64": a + b, "sub.u64": a - b, "mul.lo.u64": a * b,
            "and.b64": a & b, "or.b64": a | b, "xor.b64": a ^ b,
            "min.u64": min(a, b), "max.u64": max(a, b)}[name] & M64


def _bfi(a, b, pos, wid):
    mask = ((1 << wid) - 1) << pos
    return (((a << pos) & mask) | (b & ~mask)) & M32


def _prmt(a, b, sel):
    src = [(a >> (8 * k)) & 0xFF for k in range(4)] + [(b >> (8 * k)) & 0xFF for k in range(4)]
    res = 0
    for k in range(4):
        s = (sel >> (4 * k)) & 7
        res |= src[s] << (8 * k)
    return res & M32


def _lop3(a, b, c, table):
    a, b, c = a & M32, b & M32, c & M32
    # PTX truth-table bit order: index = a<<2 | b<<1 | c  (a is the MSB) -- verified on GPU.
    res = 0
    for bit in range(32):
        idx = ((a >> bit) & 1) << 2 | ((b >> bit) & 1) << 1 | ((c >> bit) & 1)
        if (table >> idx) & 1:
            res |= 1 << bit
    return res


def _cmp(a, b, op):
    return 1 if {"lt": a < b, "gt": a > b, "eq": a == b, "ne": a != b,
                 "ge": a >= b}[op] else 0


# ---------------------------------------------------------------------------
# Program generation (deterministic: seed -> program)
# ---------------------------------------------------------------------------
class _Program:
    def __init__(self, nr32: int, nr64: int, npred: int) -> None:
        self.nr32, self.nr64, self.npred = nr32, nr64, npred
        self.steps: List[Dict[str, Any]] = []   # each: op, dst, srcs, imm, extra
        self._last_def: Dict[str, int] = {}     # reg key -> step index
        self._next_dst_r32 = 0
        self._next_dst_r64 = 0

    # -- register helpers ----------------------------------------------------
    @staticmethod
    def r32(i): return ("r32", i)

    @staticmethod
    def r64(i): return ("r64", i)

    @staticmethod
    def pred(i): return ("pred", i)

    def add(self, op, dst, srcs, imm=None, note=""):
        self.steps.append({"op": op, "dst": dst, "srcs": srcs, "imm": imm, "note": note})
        self._last_def[dst] = len(self.steps) - 1

    def _pick_src(self, kind: str, rng: random.Random, chain_prev: Any) -> Any:
        n = {"r32": self.nr32, "r64": self.nr64, "pred": self.npred}[kind]
        # bias: 40% most-recently-defined register (long chain), 30% oldest, 30% random
        if n == 0:
            raise ValueError(f"no {kind} registers")
        r = rng.random()
        if r < 0.40 and chain_prev is not None and chain_prev[0] == kind:
            return chain_prev
        if r < 0.70 and self._last_def:
            oldest = min(self._last_def, key=self._last_def.get)
            if oldest[0] == kind:
                return oldest
        return (kind, rng.randrange(n))

    def _pick_dst32(self, rng) -> Tuple[str, int]:
        if self._next_dst_r32 < self.nr32:
            d = self.r32(self._next_dst_r32)
            self._next_dst_r32 += 1
            return d
        # overwrite a random reg (keeps pressure high)
        return self.r32(rng.randrange(self.nr32))

    def _pick_dst64(self, rng) -> Tuple[str, int]:
        if self._next_dst_r64 < self.nr64:
            d = self.r64(self._next_dst_r64)
            self._next_dst_r64 += 1
            return d
        return self.r64(rng.randrange(self.nr64))


def _gen_program(seed: int, params: Dict[str, Any]) -> _Program:
    rng = random.Random(seed)
    nr32 = max(4, int(params.get("n_regs32", 10)))
    nr64 = max(3, int(params.get("n_regs64", 6)))
    npred = max(2, int(params.get("n_preds", 4)))
    n_instr = int(params.get("n_instr", 150))
    use_rare = bool(params.get("use_rare_ops", True))
    use_imm = bool(params.get("use_immediates", True))

    prog = _Program(nr32, nr64, npred)

    # ---- setup: define all data registers (input + immediates) ---------------
    for i in range(nr32):
        if i < 4:
            prog.add("mov.b32", prog.r32(i), [("in32", i)])
        else:
            prog.add("mov.b32_i", prog.r32(i), [], rng.getrandbits(32))
    for i in range(nr64):
        if i < 2:
            prog.add("mov.u64", prog.r64(i), [("in64", i)])
        else:
            prog.add("mov.u64_i", prog.r64(i), [], rng.getrandbits(64))
    # initial predicates
    for i in range(npred):
        a = prog.r32(rng.randrange(nr32))
        b = prog.r32(rng.randrange(nr32))
        op = rng.choice(["setp.lt.u32", "setp.gt.u32", "setp.eq.u32", "setp.ne.u32"])
        prog.add(op, prog.pred(i), [a, b])

    # ---- compute phase: every instruction's result is folded into one of two
    # accumulator chains (rd0 / rd1) IMMEDIATELY, so ptxas cannot dead-code it away
    # at any -O level.  Sources are biased toward older registers so values stay live
    # long (register pressure), while the two accumulators give scheduling pressure.
    r32_ops = [o for o in _OPS_BY_DST["r32"] if o["name"] in
               ("add.u32", "sub.u32", "mul.lo.u32", "mul.hi.u32", "and.b32", "or.b32",
                "xor.b32", "min.u32", "max.u32", "popc.b32", "clz.b32", "brev.b32",
                "cvt.u32.u64", "selp.b32")]
    r64_ops = [o for o in _OPS_BY_DST["r64"] if o["name"] in
               ("add.u64", "sub.u64", "mul.lo.u64", "mul.hi.u64", "and.b64", "or.b64",
                "xor.b64", "min.u64", "max.u64", "cvt.u64.u32", "brev.b64")]
    pred_ops = [o for o in _OPS_BY_DST["pred"] if o["name"] in
                ("setp.lt.u32", "setp.gt.u32", "setp.eq.u32", "setp.ne.u32",
                 "setp.lt.u64", "setp.gt.u64", "setp.eq.u64", "and.pred", "or.pred",
                 "not.pred")]
    if use_rare:
        r32_ops += [o for o in _OPS_BY_DST["r32"] if o["name"] in
                    ("bfe.u32", "bfi.b32", "prmt.b32", "lop3.b32", "sad.u32",
                     "shf.r.wrap.b32", "popc.b64", "clz.b64", "shl.b32", "shr.u32",
                     "shl.b32_i", "shr.u32_i")]
        r64_ops += [o for o in _OPS_BY_DST["r64"] if o["name"] in
                    ("shl.b64", "shr.u64", "shl.b64_i", "shr.u64_i")]
    if use_imm:
        r32_ops += [o for o in _OPS_BY_DST["r32"] if o["name"] == "mov.b32_i"]
        r64_ops += [o for o in _OPS_BY_DST["r64"] if o["name"] == "mov.u64_i"]

    rd0, rd1 = prog.r64(0), prog.r64(1)
    scratch = prog.r64(2)
    fold_r32 = prog.r32(0)
    chain_prev = None
    for k in range(n_instr):
        kind = rng.choices(["r32", "r64", "pred"], weights=[0.55, 0.25, 0.20])[0]
        pool = {"r32": r32_ops, "r64": r64_ops, "pred": pred_ops}[kind]
        op = rng.choice(pool)
        if op["imm"] in ("shift32", "shift64"):
            value_kind = op["srcs"][0]
            value = prog._pick_src(value_kind, rng, chain_prev)
            if value_kind == "r32":
                tmp = prog.r32(rng.randrange(nr32))
                cnt_src = prog._pick_src("r32", rng, chain_prev)
                prog.add("and.b32", tmp, [cnt_src, _const_kind(31)], note="mask-shift")
            else:
                tmp = prog.r32(rng.randrange(nr32))
                cnt_src = prog._pick_src("r64", rng, chain_prev)
                prog.add("cvt.u32.u64", tmp, [cnt_src], note="shift-low32")
                prog.add("and.b32", tmp, [tmp, _const_kind(63)], note="mask-shift")
            srcs = [value, tmp]
        else:
            srcs = []
            for stype in op["srcs"]:
                if stype == "pred":
                    srcs.append(prog.pred(rng.randrange(npred)))
                else:
                    srcs.append(prog._pick_src(stype, rng, chain_prev))
        # destination: round-robin (no random overwrite -> fewer dead writes)
        if kind == "r32":
            dst = prog.r32(k % nr32)
        elif kind == "r64":
            dst = prog.r64(k % nr64)
        else:
            dst = prog.pred(k % npred)
        # immediate
        imm = None
        it = op["imm"]
        if it == "u32":
            imm = rng.getrandbits(32)
        elif it == "u64":
            imm = rng.getrandbits(64)
        elif it == "imm5":
            imm = rng.randrange(32)
        elif it == "imm6":
            imm = rng.randrange(64)
        elif it in ("bfe", "bfi"):
            # PTX requires pos + wid <= 32 (else undefined) -- enforce it strictly.
            pos = rng.randrange(0, 16)
            wid = rng.randrange(1, min(16, 32 - pos) + 1)
            imm = (pos, wid)
        elif it == "prmt":
            sel = 0
            for _k in range(4):
                sel |= rng.randrange(8) << (4 * _k)
            imm = sel
        elif it == "lop3":
            imm = rng.randrange(256)
        prog.add(op["name"], dst, srcs, imm)
        if dst[0] in ("r32", "r64"):
            chain_prev = dst

        # --- immediate fold into the accumulator (all instructions essential) ----
        if kind == "r32":
            prog.add("cvt.u64.u32", scratch, [dst])
            prog.add("xor.b64", rd0 if k % 2 == 0 else rd1, [rd0 if k % 2 == 0 else rd1, scratch])
        elif kind == "r64":
            prog.add("xor.b64", rd0 if k % 2 == 0 else rd1, [rd0 if k % 2 == 0 else rd1, dst])
        else:  # predicate -> fold via selp
            prog.add("selp.b32", fold_r32, [prog.r32(1 % nr32), prog.r32(2 % nr32), dst])
            prog.add("cvt.u64.u32", scratch, [fold_r32])
            prog.add("xor.b64", rd0 if k % 2 == 0 else rd1, [rd0 if k % 2 == 0 else rd1, scratch])
        # keep the other accumulator mixed in occasionally (independent path)
        if k % 7 == 0:
            prog.add("add.u64", rd0, [rd0, rd1])

    # final outputs
    prog.add("out", rd0, [])
    prog.add("out", rd1, [])
    return prog


def _const_kind(v: int) -> Tuple[str, int]:
    return ("imm32_const", v)


# ---------------------------------------------------------------------------
# Emit the CUDA kernel from a program.
# ---------------------------------------------------------------------------
def _reg_name(reg) -> str:
    kind, idx = reg
    return {"r32": "%%xr", "r64": "%%xd", "pred": "%%xp"}[kind] + str(idx)


def _src_name(src) -> str:
    kind, idx = src
    if kind == "in32":
        return f"%{2 + idx}"
    if kind == "in64":
        return f"%{6 + idx}"
    if kind == "imm32_const":
        return f"{idx}"
    return _reg_name(src)


def _emit_asm(prog: _Program) -> str:
    lines = [
        f".reg .b32 %%xr<{prog.nr32}>;",
        f".reg .b64 %%xd<{prog.nr64}>;",
        f".reg .pred %%xp<{prog.npred}>;",
    ]
    for step in prog.steps:
        if step["op"] == "out":
            lines.append(f"mov.u64 %{0 if step['dst'][1] == 0 else 1}, {_reg_name(step['dst'])};")
            continue
        op = next(o for o in OPS if o["name"] == step["op"])
        dst = _reg_name(step["dst"])
        srcs = [_src_name(s) for s in step["srcs"]]
        imm = step["imm"]
        if op["imm"] == "bfe" or op["imm"] == "bfi":
            line = op["emit"](dst, srcs, imm)
        else:
            line = op["emit"](dst, srcs) if imm is None else op["emit"](dst, srcs, imm)
        lines.append(line)
    # escape newlines/tabs so the asm string stays a single C string literal
    return "\\n\\t".join(lines)


def _emit_cuda(prog: _Program) -> str:
    asm_body = _emit_asm(prog)
    return f"""\
// strategy=inline_ptx  (generated; do not edit)
extern "C" __global__ void test_kernel(
    const unsigned long long* __restrict__ input,
    unsigned long long* __restrict__ output,
    unsigned int input_words,
    unsigned long long seed)
{{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // runtime operands feeding the inline PTX block
    unsigned long long w0 = input[(tid * 3u) % input_words] ^ seed;
    unsigned long long w1 = input[(tid * 5u + 1u) % input_words] ^ (seed >> 11u);
    unsigned long long w2 = input[(tid * 7u + 2u) % input_words] ^ (seed >> 29u);
    unsigned long long w3 = input[(tid * 11u + 3u) % input_words] ^ (seed >> 37u);
    unsigned int a0 = (unsigned int)(w0 ^ (w0 >> 32u));
    unsigned int a1 = (unsigned int)(w1 ^ (w1 >> 32u));
    unsigned int a2 = (unsigned int)(w2) ^ (unsigned int)(seed >> 47u);
    unsigned int a3 = (unsigned int)(w3);
    unsigned long long b0 = input[(tid * 13u + 4u) % input_words] ^ (seed * 3u);
    unsigned long long b1 = input[(tid * 17u + 5u) % input_words] ^ (seed >> 13u);

    unsigned long long o0 = 0ULL, o1 = 0ULL;
    asm volatile(
        "{asm_body}"
        : "=l"(o0), "=l"(o1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b0), "l"(b1)
        : "memory");

    output[tid * 2u] = o0;
    output[tid * 2u + 1u] = o1;
}}
"""


# ---------------------------------------------------------------------------
# CPU oracle: simulate the same program.
# ---------------------------------------------------------------------------
def _simulate(prog: _Program, a0, a1, a2, a3, b0, b1) -> Tuple[int, int]:
    r32 = [0] * prog.nr32
    r64 = [0] * prog.nr64
    preds = [0] * prog.npred
    for step in prog.steps:
        op = step["op"]
        if op == "out":
            v = r64[step["dst"][1]]
            if step["dst"][1] == 0:
                o0 = v
            else:
                o1 = v
            continue
        src_vals = []
        for s in step["srcs"]:
            kind, idx = s
            if kind == "in32":
                src_vals.append([a0, a1, a2, a3][idx])
            elif kind == "in64":
                src_vals.append([b0, b1][idx])
            elif kind == "imm32_const":
                src_vals.append(idx)
            elif kind == "r32":
                src_vals.append(r32[idx])
            elif kind == "r64":
                src_vals.append(r64[idx])
            elif kind == "pred":
                src_vals.append(preds[idx])
        o = next(x for x in OPS if x["name"] == op)
        result = o["mirror"](src_vals, step["imm"]) if step["imm"] is not None else o["mirror"](src_vals)
        dk, di = step["dst"]
        if dk == "r32":
            r32[di] = result & M32
        elif dk == "r64":
            r64[di] = result & M64
        else:
            preds[di] = 1 if result else 0
    return o0, o1


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class InlinePtx(Strategy):
    name = "inline_ptx"
    target_opcodes = ["prmt", "shf", "bfe", "bfi", "lop3", "sad", "mul.hi",
                      "popc", "clz", "brev", "selp", "setp"]

    def _fmt_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "n_instr": int(p.get("n_instr", 150)),
            "n_regs32": int(p.get("n_regs32", 10)),
            "n_regs64": int(p.get("n_regs64", 6)),
            "n_preds": int(p.get("n_preds", 4)),
            "use_rare_ops": bool(p.get("use_rare_ops", True)),
            "use_immediates": bool(p.get("use_immediates", True)),
        }

    def _program(self, seed: int, params: Dict[str, Any]) -> _Program:
        return _gen_program(seed, params)

    def generate(self, config_seed: int, test_seed: int, params: Dict[str, Any],
                 gen_version: str, target_sm: str) -> GeneratedCase:
        p = self._fmt_params(params)
        rng = self._new_rng(config_seed, test_seed)
        input_words = 256
        words = make_input_words(rng, input_words, 8)
        seed = rng.getrandbits(64)
        prog = self._program(seed, p)
        source = _emit_cuda(prog)
        grid, block = [1], [64]
        owpt = 2
        output_words = grid[0] * block[0] * owpt
        input_bytes = pack_words(words)
        expected = self.cpu_oracle(input_words, seed, p, input_bytes, grid, block, owpt)
        return GeneratedCase(
            test_id=f"inline_ptx__{config_seed:x}__{test_seed:x}",
            config_seed=config_seed, test_seed=test_seed, strategy=self.name, params=p,
            generator_version=gen_version, kernel_seed=seed, source=source,
            input_bytes=input_bytes, expected_bytes=expected,
            grid=grid, block=block, input_words=input_words, output_words=output_words,
            output_words_per_thread=owpt, target_sm=target_sm,
            cpu_oracle=self.name, comparison_policy="exact_u64",
            cuda_features=[target_sm, "inline_ptx"],
        )

    def cpu_oracle(self, input_words: int, seed: int, params: Dict[str, Any],
                   input_bytes: bytes, grid: List[int], block: List[int],
                   output_words_per_thread: int) -> bytes:
        p = self._fmt_params(params)
        prog = self._program(seed, p)
        words = unpack_words(input_bytes)
        total = grid[0] * block[0]
        out: List[int] = []
        for tid in range(total):
            w0 = words[(tid * 3) % input_words] ^ seed
            w1 = words[(tid * 5 + 1) % input_words] ^ (seed >> 11)
            w2 = words[(tid * 7 + 2) % input_words] ^ (seed >> 29)
            w3 = words[(tid * 11 + 3) % input_words] ^ (seed >> 37)
            a0 = (w0 ^ (w0 >> 32)) & M32
            a1 = (w1 ^ (w1 >> 32)) & M32
            a2 = (w2 & M32) ^ ((seed >> 47) & M32)
            a3 = w3 & M32
            b0 = words[(tid * 13 + 4) % input_words] ^ (seed * 3)
            b1 = words[(tid * 17 + 5) % input_words] ^ (seed >> 13)
            o0, o1 = _simulate(prog, a0, a1, a2, a3, b0 & M64, b1 & M64)
            out.append(o0 & M64)
            out.append(o1 & M64)
        return pack_words(out)

    def describe_pressure(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instructions": p.get("n_instr"),
            "regs32": p.get("n_regs32"), "regs64": p.get("n_regs64"),
            "preds": p.get("n_preds"), "rare_ops": p.get("use_rare_ops"),
        }
