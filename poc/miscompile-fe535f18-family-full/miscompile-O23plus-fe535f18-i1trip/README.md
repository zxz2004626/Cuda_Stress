# miscompile-O23plus-fe535f18-i1trip

Follow-up variant discovered while trying to further reduce
`poc/miscompile-O1only-fe535f18/reduced.cu`.

Compared with the O1-only sibling, this changes only the inner mixer loop bound:

```cpp
for (u i_1 = 0; i_1 < 3; ++i_1)
```

The change does not preserve the O1-only differential. Instead, CUDA 13.3
`ptxas` moves to an O2/O3 wrong-result class.

## Expected Result

Observed on CUDA 13.3 targeting `sm_86`:

| opt | status | output |
| --- | --- | --- |
| O0 | CORRECT | 4123129483 |
| O1 | CORRECT | 4123129483 |
| O2 | WRONG | 3429195403 |
| O3 | WRONG | 3429195403 |

This is saved as a separate research candidate because it demonstrates how a
small live-range/control-shape perturbation can move the same backend issue
from O1-only into the O2/O3 ptxas pipeline.
