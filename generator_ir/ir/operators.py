"""Operator vocabularies for the gpuperf IR.

Kept deliberately small and closed: every emitter and mutation can rely on
these exact members and their symbol/semantic tables.
"""

from __future__ import annotations

from enum import Enum, auto


class Arith(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    BAND = auto()
    BOR = auto()
    BXOR = auto()
    SHL = auto()
    SHR = auto()

    @property
    def symbol(self) -> str:
        return _ARITH_SYM[self]




_ARITH_SYM = {
    Arith.ADD: "+", Arith.SUB: "-", Arith.MUL: "*", Arith.DIV: "/",
    Arith.MOD: "%", Arith.BAND: "&", Arith.BOR: "|", Arith.BXOR: "^",
    Arith.SHL: "<<", Arith.SHR: ">>",
}


class Cmp(Enum):
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    @property
    def symbol(self) -> str:
        return _CMP_SYM[self]

    def negated(self) -> "Cmp":
        return _CMP_NEG[self]

    def swapped(self) -> "Cmp":
        """Operator for the swapped-operand form (a < b -> b > a)."""
        return _CMP_SWAP[self]


_CMP_SYM = {
    Cmp.EQ: "==", Cmp.NE: "!=", Cmp.LT: "<",
    Cmp.GT: ">", Cmp.LE: "<=", Cmp.GE: ">=",
}
_CMP_NEG = {
    Cmp.EQ: Cmp.NE, Cmp.NE: Cmp.EQ, Cmp.LT: Cmp.GE,
    Cmp.GE: Cmp.LT, Cmp.GT: Cmp.LE, Cmp.LE: Cmp.GT,
}
_CMP_SWAP = {
    Cmp.EQ: Cmp.EQ, Cmp.NE: Cmp.NE, Cmp.LT: Cmp.GT,
    Cmp.GT: Cmp.LT, Cmp.LE: Cmp.GE, Cmp.GE: Cmp.LE,
}


class Logic(Enum):
    AND = auto()
    OR = auto()

    @property
    def symbol(self) -> str:
        return "&&" if self is Logic.AND else "||"


class UnaOp(Enum):
    NEG = auto()
    LNOT = auto()
    BNOT = auto()

    @property
    def symbol(self) -> str:
        return {UnaOp.NEG: "-", UnaOp.LNOT: "!", UnaOp.BNOT: "~"}[self]


class Compound(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    BAND = auto()
    BOR = auto()
    BXOR = auto()
    SHL = auto()
    SHR = auto()

    @property
    def symbol(self) -> str:
        return _COMPOUND_SYM[self] + "="

    def to_arith(self) -> Arith:
        return _COMPOUND_TO_ARITH[self]


_COMPOUND_SYM = {
    Compound.ADD: "+", Compound.SUB: "-", Compound.MUL: "*",
    Compound.DIV: "/", Compound.MOD: "%", Compound.BAND: "&",
    Compound.BOR: "|", Compound.BXOR: "^", Compound.SHL: "<<",
    Compound.SHR: ">>",
}
_COMPOUND_TO_ARITH = {
    Compound.ADD: Arith.ADD, Compound.SUB: Arith.SUB, Compound.MUL: Arith.MUL,
    Compound.DIV: Arith.DIV, Compound.MOD: Arith.MOD, Compound.BAND: Arith.BAND,
    Compound.BOR: Arith.BOR, Compound.BXOR: Arith.BXOR,
    Compound.SHL: Arith.SHL, Compound.SHR: Arith.SHR,
}


class IncDec(Enum):
    INC = auto()
    DEC = auto()

    @property
    def symbol(self) -> str:
        return "++" if self is IncDec.INC else "--"


class SafeOp(Enum):
    """UB-guarded operators (see emit layer for expansion).

    SDIV/SMOD: guard zero divisor and the signed min/-1 overflow.
    SSHL/SSHR: guard out-of-range shift counts.
    SADD/SSUB/SMUL: evaluate through unsigned wraparound so signed overflow
    is defined behavior.
    """
    SDIV = auto()
    SMOD = auto()
    SSHL = auto()
    SSHR = auto()
    SADD = auto()
    SSUB = auto()
    SMUL = auto()
