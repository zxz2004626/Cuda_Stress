# C04: ptxas -O1+ miscompiles cicc-11.3 PTX on CUDA 11.1..13.0, fixed in 13.3 (majority-wrong differential)

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C04**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.1/O0-O1-O2 + 11.3/O0 + 11.6/O0 + 11.8/O0 + 12.2/O0 + 12.5/O0 + 12.8/O0 + 13.0/O0 + 13.3/O0-O1-O2（错侧=多数派 11.3–13.0 的 O1/O2）`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause

ptxas -O1+ 优化 bug：同一（cicc 11.3 生成的）PTX 在 ptxas 11.1/11.3/11.6/11.8/12.2/12.5/12.8/13.0 的 -O1/-O2/-O3 下全部编错（-O0 一律正确），ptxas 13.3 修复（-O1 也正确）。nvcc 13.3 全档正确，即 NVIDIA 已在 CUDA 13.3 修复此回归；本 POC 用 cuda-11.3 复现。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Fixed in: CUDA **13.3**.

## Files

| file | description |
| --- | --- |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums. |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C04
make            # builds with CUDA 13.3 ptxas -O0/-O1/-O2, prints each checksum
make clean
```

Use a different toolkit:

```bash
make NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.0/bin/nvcc
```

## Expected result (measured on RTX 3090, sm_86)

| toolkit | -Xptxas | verdict | checksum |
| --- | --- | --- | --- |
| CUDA 11.3 | -O0 | CORRECT | 13009286742761524257 |
| CUDA 11.3 | -O1 | WRONG | 6759641714795783782 |
| CUDA 11.3 | -O2 | WRONG | 6759641714795783782 |
| CUDA 11.3 | -O3 | WRONG | 6759641714795783782 |
| CUDA 13.3 | -O1 | CORRECT | 13009286742761524257 |

- source-correct checksum (GOOD side): `13009286742761524257`
- miscompiled checksum (BAD side): `6759641714795783782`

## Provenance

- differential class: `C04` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C04_m0` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
