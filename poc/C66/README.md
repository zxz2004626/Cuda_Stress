# C66: ptxas -O1-only miscompile since 11.8, unfixed through 13.3

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C66**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.8/O1 + 12.2/O1 + 12.5/O1 + 12.8/O1 + 13.0/O1 + 13.3/O1`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：源码无 UB（重申并维持）。ptxas -O1 专有的指令调度错误破坏
`v_24 = (88<<24)^(13>>24)^0xFFFFFFFFFFFFFFFF | funnel(input,255)` 项：
22369/22528 个线程输出错误，XOR 位型随线程变化（bit 11/12/13/14/23/27 组合），
与"按 lane 变化的量（s = input[tid] & 31）经过错误调度/uniform 通路"的形态一致。
插桩实测 `-O1` 构建的 `v_24 = 0xFFFFFF60720057FF` 等（正确 `0xFFFFFFFFF7FFFFFF`，
tid=0），常数基掩码与漏斗位均错位；`v_31 = 0xFFFFFFFF` 正确。**

### 实测（RTX 3090, sm_86，含 `-Xptxas` 全级别重测）

- ptxas `-O0/-O2/-O3`（11.1–13.3 各版本）与 nvcc 11.5 默认：`1727017707039551184`
  （= Python 标准语义模型，逐位一致）
- ptxas `-O1`（11.8 起：11.8/12.2/12.5/12.8/13.0/13.2/13.3）：`4501211274617234520`
- 三级 `-Xptxas` 的 PTX md5 完全一致 → 错误 100% 在 ptxas -O1 的 SASS 生成。

---

## Root cause (preliminary, from archived trigger matrix)

ptxas -O1 专有 bug：归档矩阵 11.8 起的 O1 整列错（O0/O2/O3 正确），跨 12.2 未修（最近的 -O1-only 家族 ffsll 已在 12.2 修复，故为不同缺陷）。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Status: **unfixed** as of CUDA 13.3.

## Minimal core (`min.cu`, 2026-09-21 reduction)

`test.cu` was further reduced to `min.cu` (1 block × 8 threads, `n = 8`,
no `__device__` arrays, no `__ldg`, no `__brev`; only
`__byte_perm` + one runtime 64-bit `ull` division survive as the "hard" ops).

```bash
make min            # builds min_O0 / min_O1 with CUDA 13.3 and prints both checksums
```

Measured (RTX 3090, sm_86, this exact file — renaming the kernel or moving
statements can flip the trigger, the bug is scheduling-order sensitive):

| toolkit | ptxas | checksum | verdict |
| --- | --- | --- | --- |
| 11.8 / 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O0 | `16991622330266937562` | CORRECT |
| 11.8 / 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O1 | `15140686756037123777` | WRONG |
| 11.8 / 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O2 | `16991622330266937562` | CORRECT |
| 13.3 | -O3 | `16991622330266937562` | CORRECT |

ptxas input is bit-identical across -Xptxas levels (single PTX, md5
`1d3c35454300f213b50e89a403a79806`) → defect is 100 % in ptxas -O1 SASS
generation/scheduling. compute-sanitizer memcheck / initcheck / racecheck: 0
errors on the -O1 build.

### Value-level root cause (per-thread ground truth)

A Python source-semantics model matches the O0/O2/O3 outputs per thread. With
`blockDim=8, gridDim=1, n=8, var_2=3, input=[16..23]`:

- every intermediate except `v_31` is compiled correctly at -O1 (dumping any
  single term as the output yields identical checksums at all levels);
- in the full kernel, -O1 computes `v_31` as `0xFFFFFE01` instead of the
  correct `1` (`(uint)(0xFFFFFFFFFFFFFFFF / (v_11 ^ 1))`,
  `v_11 ^ 1 = 0xFFFFFFFFFFFFFFE6`); the wrong value is constant across
  threads, so the final `v_31 ^ (v_35 + (uint)v_24)` XORs the constant mask
  `0xFFFFFE00` into every output (`O1_out = O2_out ^ 0xFFFFFE00`).

SASS (13.3): the 64-bit unsigned division by the runtime value compiles to an
out-of-line helper reached via `CALL.REL.NOINC 0x6c0`. At -O1 the divisor
high word is built as `IADD3 R3, R7, UR5` (UR5 comes from the *uniform*
datapath, `UIADD3 UR5, URZ, -UR5, URZ` = `-blockDim.x`, originally materialised
for v_1's wrap-around modulo), and the uniform `(gridDim.x << (blockDim.x&31))`
lives in UR registers (`USHF.L.U32`) next to it — the uniform/per-lane
argument marshalling across the division helper call is what corrupts the
quotient's upper bits at -O1.

Reduction evidence: ~90 statement-level probe variants were compiled and
compared with an oracle requiring `ptxas -O0 == -O2 == -O3 != -O1` plus
deterministic repeat runs; each remaining construct in `min.cu` is
load-bearing (removal/rewrite loses the differential). Several
individually-inert rewrites (e.g. left shift→mul combined with `^`→`+` in
v_24) jointly destroy the trigger — do not "clean up" the expression shapes.

## Files

| file | description |
| --- | --- |
| `min.cu` | Minimal core reproducer (this reduction; see section above). |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums (`make min` builds min.cu). |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C66
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
| CUDA 13.3 | -O0 | CORRECT | 1727017707039551184 |
| CUDA 13.3 | -O1 | WRONG | 4501211274617234520 |
| CUDA 13.3 | -O2 | CORRECT | 1727017707039551184 |
| CUDA 13.3 | -O3 | CORRECT | 1727017707039551184 |

- source-correct checksum (GOOD side): `1727017707039551184`
- miscompiled checksum (BAD side): `4501211274617234520`

## Provenance

- differential class: `C66` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C66_m0` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
