# C68: ptxas -O2/-O3 constant-folding bug (XOR 0x3b), triggerable since 12.2 shape, unfixed

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C68**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`12.2/O2 + 12.5/O2 + 12.8/O2 + 13.0/O2 + 13.3/O2`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：源码无 UB。ptxas -O2/-O3 对"按常数 127 取模"的魔数降级错误：
`v_38 = ((v_31 & 256) % 127) % ((v_31 - 1) | 1)` 的内层 `% 127` 被降级为
等效于 `% 199`（`256 % 127 = 2` 被算成 `256 % 199 = 57`），
全 16384 线程输出整体差常量 0x3B（= 2 ^ 57）。**

### 实测（RTX 3090, sm_86，含 `-Xptxas` 全级别重测）

- 标准/源语义（g++ 主机参考 = Python 模拟 = ptxas `-O0/-O1`）：`15618583459280906833`
- ptxas `-O2/-O3`（12.6 与 13.3 的默认构建）：`11963299404108113300`
- 逐线程 diff：全部 16384 线程 XOR = 恰为 0x3B（常量）；
- 探针实测：`v_9`（0xE90FB376）与 `v_0` 两级完全一致，
  仅 `v_38`：正确 2 → 错误 0x39（57）；
- 求解器在 K ∈ [2, 4096) 中唯一解出 `% 199` 可复现全部错误输出；
- 消融实验（`v_9` 换成常量）后 -O2 的 `v_38` 恢复正确 → 错误依赖完整内核的
  指令调度（`% 127` 魔数序列与 `v_31` 的运行时 64 位除法链交错时被重排破坏）。

### PTX 证据

三级 `-Xptxas` 的 PTX md5 完全一致，且为干净的 `rem.u32 %r15, %r14, %r13`
语义操作（含 `% (tid|1)`、`v_22`、`v_38` 的两级 rem）→ 错误 100% 在
ptxas -O2/-O3 的 SASS 生成。

---

## Root cause (preliminary, from archived trigger matrix)

ptxas -O2/-O3 优化 bug（12.2 形状起可触发，未修）：lane 无关常量项错，全 lane 恒差 XOR 0x3b。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Status: **unfixed** as of CUDA 13.3.

## Minimal core (`min.cu`, 2026-09-21 reduction)

`test.cu` was further reduced to `min.cu` (1 block × 8 threads, n = 16384 → 8
threads total, no dynamic shared memory).

```bash
make min            # builds min_O0 / min_O2 with CUDA 13.3 and prints both checksums
```

The lane-uniform miscompiled constant term is `v_38`, fed by
`v_31 = (u64)(u32)( ~(u64)blockIdx.x / (((12784972066293982486ull / (u64)blockDim.x) | 1)) )`
with `blockIdx.x = 0`, combined through the three-XOR output tree
`input ^ v_0 ^ v_9 ^ v_38`. Per-thread ground truth: every output differs from
the correct value by a **constant XOR** (0x3b in test.cu's original
configuration, 0xa6 for min.cu's reduced configuration — the constant depends
on the surrounding context, the "all lanes, constant XOR" signature does not).

Measured for `min.cu` (RTX 3090, sm_86, this exact file):

| toolkit | ptxas | checksum | verdict |
| --- | --- | --- | --- |
| 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O0 | `13369739290192476610` | CORRECT |
| 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O1 | `13369739290192476610` | CORRECT |
| 12.2 / 12.5 / 12.8 / 13.0 / 13.1 / 13.2 / 13.3 | -O2/-O3 | `13609370734393575647` | WRONG |
| 11.8 (negative control) | -O0/-O2 | `13369739290192476610` | CORRECT |
| 11.5 (negative control) | -O0/-O1/-O2/-O3 | `13369739290192476610` | CORRECT |

compute-sanitizer memcheck / initcheck / racecheck on the -O2 build: 0 errors.

Reduction evidence: single-term and two-term outputs (input^v0^v9,
input^v0^v38, input^v9^v38, each term alone) are all computed identically at
-O1 and -O2 — the misfold only appears with the full three-XOR chain.
Required context (removal/rewrite kills the trigger): the `v_28` definition
(value 0, definition load-bearing), the `v_10` definition (value 0, reads
`n`), the `v_22 != 0` branch with empty body, the dead second XOR term of
`v_31`, and the guarded-op macro shapes.

## Files

| file | description |
| --- | --- |
| `min.cu` | Minimal core reproducer (this reduction; see section above). |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums (`make min` builds min.cu). |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C68
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
| CUDA 12.2 | -O0 | CORRECT | 15618583459280906833 |
| CUDA 12.2 | -O1 | CORRECT | 15618583459280906833 |
| CUDA 12.2 | -O2 | WRONG | 11963299404108113300 |
| CUDA 12.2 | -O3 | WRONG | 11963299404108113300 |

- source-correct checksum (GOOD side): `15618583459280906833`
- miscompiled checksum (BAD side): `11963299404108113300`

## Provenance

- differential class: `C68` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C68` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
