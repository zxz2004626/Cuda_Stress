"""Weave-mode block families ported from perffuzzer's codegen generators.

Importing this package registers every block in the submodules below with
``generator_ir.gen.weave_handlers.REGISTRY`` (side-effect registration, mirroring
perffuzzer's ``generator/codegen/__init__.py`` auto-discovery).
"""

from generator_ir.gen.weave_blocks import arith as arith  # noqa: F401
from generator_ir.gen.weave_blocks import control as control  # noqa: F401
from generator_ir.gen.weave_blocks import narrowint as narrowint  # noqa: F401
from generator_ir.gen.weave_blocks import struct_ as struct_  # noqa: F401
from generator_ir.gen.weave_blocks import fp as fp  # noqa: F401
from generator_ir.gen.weave_blocks import fp_branch as fp_branch  # noqa: F401
from generator_ir.gen.weave_blocks import fp_lib as fp_lib  # noqa: F401
from generator_ir.gen.weave_blocks import memory as memory  # noqa: F401
from generator_ir.gen.weave_blocks import intrinsic as intrinsic  # noqa: F401
from generator_ir.gen.weave_blocks import sync as sync  # noqa: F401
from generator_ir.gen.weave_blocks import wmma as wmma  # noqa: F401
from generator_ir.gen.weave_blocks import yield_ as yield_  # noqa: F401
from generator_ir.gen.weave_blocks import template as template  # noqa: F401

# Feature-coverage blocks default to weight 0 unless --enable-feature-gens is
# parsed (mirrors perffuzzer generator/codegen/__init__.py's import-time
# ``_apply_feature_gen_weight(enabled=False)``).  Must run AFTER the submodule
# imports above so every feature block is already registered; blocks that
# omitted an explicit ``weight=`` otherwise default to 1.0.
from generator_ir.gen.weave_handlers import apply_feature_gen_weight
apply_feature_gen_weight(enabled=False)
