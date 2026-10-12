# C50: cicc 13.3 miscompile with mixed O0-column matrix

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C50**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.1/O0-O1-O2 + 11.3/O0-O1-O2 + 11.6/O0 + 11.8/O0 + 12.2/O0 + 12.5/O0 + 12.8/O0 + 13.0/O0 + 13.3/O0-O1-O2`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：本 POC 源码有 UB——`eg_la_0[8]` 未初始化就被读取
（第 36-37 行，ISO C++ 对不确定值做左值到右值转换为 UB）。
但该 UB 被"双重异或"数学抵消（`eg_acc_0` 在第 38 与 62 行各异或一次），
实测把 `eg_acc_0` 固定为任意常量（含全零的无 UB 变体）输出完全不变，
因此它不是跨版本差异的原因。真正的差异原因与 C32 同族：
ptxas 对 `(short)(-x)` 回绕到 INT16_MIN 的截断丢弃
（`v_29 = +32768` 未截断，`(short)1 > v_29` 误判为 false）。**

### 触发点

全 32768 线程中恰有 tid=32736（`(31 ^ ~tid) & 0xFFFF == 0x8000`）使命题
`(short)1 > v_29` 的取值在正确/错误实现间翻转（v29 = -32768 vs +32768），
经 `v_43 → cse9 → v_21` 链使该线程输出差 1 bit。

### 实测（RTX 3090, sm_86，重测确认）

- 标准/源语义（Python 模拟 = g++ 主机参考 = nvcc `-G`，11.5/12.6/13.0）：`8475015328089267462`
- nvcc 11.5/12.6/13.0 的 `-O0/-O1/-O2/-O3`/`--use_fast_math`：`14033049871667513618`
- 逐线程 diff：32768 线程中仅 tid=32736 差 1 bit。

### PTX / SASS 证据

- PTX 正确：`neg.s16` + `setp.lt.s16`；
- SASS（错误）：`PRMT sext16` → `IMAD.MOV R4, -R4`（32 位取负）→
  `ISETP.GE R4, 0x1`——与 C32 逐字同款。

---

## Root cause (preliminary, from archived trigger matrix)

cicc 13.3 前端错（混合矩阵：13.3 全档与多版本 O0 列同落错值）。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Status: **unfixed** as of CUDA 13.3.

## Files

| file | description |
| --- | --- |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums. |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C50
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
| CUDA 13.3 | -O0 | WRONG | 8475015328089267462 |
| CUDA 13.3 | -O1 | WRONG | 8475015328089267462 |
| CUDA 13.0 | -O0 | WRONG | 8475015328089267462 |
| CUDA 13.0 | -O1 | CORRECT | 14033049871667513618 |

- source-correct checksum (GOOD side): `14033049871667513618`
- miscompiled checksum (BAD side): `8475015328089267462`

## Provenance

- differential class: `C50` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C50_m0` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
