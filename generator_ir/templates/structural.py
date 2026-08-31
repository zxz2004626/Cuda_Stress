"""Re-export hub: structural templates split into 1D + 2D."""
from generator_ir.templates.templates_1d import (  # noqa
    ElementwiseTemplate, ReductionTemplate, StencilTemplate,
)
from generator_ir.templates.templates_2d import (  # noqa
    MatmulTemplate, ScanTemplate, HistogramTemplate, TransposeTemplate,
)
