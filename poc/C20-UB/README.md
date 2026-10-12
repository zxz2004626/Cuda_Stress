# C20: version-uniform O0-column divergence: 13.3-cicc PTX is wrong on every ptxas except 11.1

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C20**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.1/O0 + 11.3/O0 + 11.6/O0 + 11.8/O0 + 12.2/O0 + 12.5/O0 + 12.8/O0 + 13.0/O0 + 13.3/O0（错侧=全版本 O0 列）`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：源码级 UB（全局内存写写竞争），不是编译器 bug。** 上述初步归因
（"O0 列分歧"）是竞争结果对代码排布敏感所造成的表象。

### 根因

`tid` 被生成为常量 `unsigned int tid = 0u;`（第 25 行，未按 threadIdx 展开），
导致连锁效应：

1. 守卫 `if (tid < n)`（第 63 行）对全部 64×256 = 16384 个线程恒真；
2. 写索引 `output[0u]`（第 64 行）为常量 0 → **所有线程无同步地写同一地址**；
3. block 0 计算 `v_31` 的乘积为 0，`_fv = input[0]`；block 1–63 的 `_fv = input[0] ^ 0x8FD0`
   —— 两种值竞争，最后落点的 store 胜出。

宏层面的"UB-proof"算术防护（除零/移位/溢出）全部有效，唯独没有覆盖这个 store。

### 实测（RTX 3090, sm_86）

两个合法竞争结果对应的 checksum（Python 精确计算）：
- block 0 胜：`14485734784133096044`
- block 1–63 胜：`2721586278693748655`（实测值；63:1 的写入方压倒性获胜，故极其稳定）
- 单 block 放行实验（`blockIdx.x == 0 / 1 / 63`）证实每个 block 的写入值与上表一致，
  三个版本的编译产物均无 miscompile。

### PTX / SASS 证据

- PTX：`ld.global.nc.u32`（input[0]）+ 64 条 `atom.global.xor.b32 [%rd1], 0`
  （`__brevll(0)=0`，恒等异或）+ `st.global.u32 [%rd1]`，无任何 fence/同步；
- SASS（sm_86）：原子操作被合法化为每 warp 仅 lane 31 执行的
  `RED.E.XOR.STRONG.GPU [R4.64], R3`（×64），最终 `STG.E [R4.64], R7` 为弱序 store，
  跨 block 无顺序保证。

### 修复建议

`tid` 改为真实 `threadIdx` 展开，或写索引包含 block/thread 维度（`output[blockIdx.x]`），
或仅允许一个线程写。compute-sanitizer `racecheck` 不覆盖全局内存竞争，
故先期流水线的 "clean under racecheck" 记录与此结论不矛盾。


---

## Root cause (preliminary, from archived trigger matrix)

cicc 13.3 前端（版本无关形态）：化简核由 cicc 13.3 编译出的 PTX 语义本身错误——除 ptxas 11.1/11.3 外，其余所有版本的 ptxas 在任何优化档都算出错值；仅 cuda-11.1 工具链给出正确值。归档 27 格矩阵为『全版本 O0 列错』，属罕见的版本无关型前端/后端一致偏离。

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
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C20
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
| CUDA 13.3 | -O0 | WRONG | 2721586278693748655 |
| CUDA 13.3 | -O1 | WRONG | 2721586278693748655 |
| CUDA 13.3 | -O2 | WRONG | 2721586278693748655 |
| CUDA 11.1 | -O1 | CORRECT | 12073729951165030900 |
| CUDA 11.3 | -O1 | WRONG | 2721586278693748655 |
| CUDA 12.8 | -O1 | WRONG | 2721586278693748655 |

- source-correct checksum (GOOD side): `12073729951165030900`
- miscompiled checksum (BAD side): `2721586278693748655`

## Provenance

- differential class: `C20` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C20_m1` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
