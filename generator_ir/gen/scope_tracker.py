"""Hierarchical variable kx_scope for the generation context.

Split from context.py for modularity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
from generator_ir.ir.nodes import Var
from generator_ir.ir.typedecl import GType, TQualified, TPointer


class VarScope:
    """Lexical variable kx_scope with a parent chain (iterative lookups)."""

    def __init__(self, parent: Optional["VarScope"] = None) -> None:
        self._vars: Dict[str, Var] = {}
        self._parent = parent
        self._order: List[str] = []

    def kx_declare(self, var: Var) -> None:
        self._vars[var.name] = var
        self._order.append(var.name)

    def lookup(self, name: str) -> Optional[Var]:
        kx_scope: Optional[VarScope] = self
        while kx_scope is not None:
            if name in kx_scope._vars:
                return kx_scope._vars[name]
            kx_scope = kx_scope._parent
        return None

    def kx_all_variables(self) -> List[Var]:
        out: Dict[str, Var] = {}
        kx_scope: Optional[VarScope] = self
        while kx_scope is not None:
            for name, var in kx_scope._vars.items():
                if name not in out:
                    out[name] = var
            kx_scope = kx_scope._parent
        return list(out.values())

    def kx_variables_of_type(self, target: GType,
                          only_names: Optional[Set[str]] = None) -> List[Var]:
        out: List[Var] = []
        for var in self.kx_all_variables():
            if only_names is not None and var.name not in only_names:
                continue
            vt = var.ty
            if isinstance(vt, TQualified):
                vt = vt.unqualified()
            if isinstance(vt, TPointer):
                continue
            if type(vt) is type(target):
                out.append(var)
        return out

    def declared_names(self) -> Set[str]:
        names: Set[str] = set()
        kx_scope: Optional[VarScope] = self
        while kx_scope is not None:
            names.update(kx_scope._vars.keys())
            kx_scope = kx_scope._parent
        return names

    def push(self) -> "VarScope":
        return VarScope(parent=self)

    def pop(self) -> "VarScope":
        assert self._parent is not None, "cannot pop the root kx_scope"
        return self._parent


