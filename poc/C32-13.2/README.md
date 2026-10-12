# C32: cicc 13.3 miscompile: wrong at every ptxas/-O level; mixed O0-column matrix

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C32**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.1/O0 + 11.3/O0 + 11.6/O0 + 11.8/O0 + 12.2/O0 + 12.5/O0 + 12.8/O0 + 13.0/O0 + 13.3/O0-O1-O2`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：源码无 UB。ptxas 把 `neg.s16` + 16 位有符号比较降级为
"32 位取负 + 32 位比较"，丢弃了取负结果必须回绕到 16 位的截断，
导致 INT16_MIN（0x8000）角上的比较结果翻转。**

### 根因与触发点

`v_25 = (s8)(-(31 ^ ~tid & 0xFFFF))`。当低 16 位恰为 `0x8000` 时，
取负回绕仍为 `-32768`；而 ptxas 生成的 SASS 在 32 位上取负得到 `+32768`，
随后的有符号比较 `v_25 > 13569` 因此由 false 翻转为 true。
全部 16384 线程中恰有一个（tid=1083，`input = 0x54b00004`）命中此角落。

### 实测（RTX 3090, sm_86，重测确认）

- 标准/源语义（Python 模拟 = g++ 主机参考 = nvcc `-G`，11.5 与 13.0）：`10138864006168268353`
- nvcc 11.5/12.6/13.0 的 `-O0/-O2/-O3`：`433407214484750329`

### PTX / SASS 证据

- PTX（-O0 与 -O3 完全一致，语义正确）：`bfe.s32`（24 位符号扩展）
  → `neg.s16`（模回绕）→ `setp.gt.s16`；
- SASS（错误）：`PRMT …, 0x9910`（sext16）→ `IMAD.MOV R3, -R3`（32 位取负，
  0xFFFF8000 → 0x00008000）→ `ISETP.GE/GT R3, 0x3501`（32768 >= 1 → true）；
- 同一 ptxas 在 -G 下生成正确的比较链。

### 家族

与 C50 同族（`(s8)/(short)` 取负回绕截断丢弃），且与已归档的
`miscompile-O1plus-u16-signext0/1`（neg.s16 后 PRMT 再符号扩展遗漏，fixed 13.3）
属同一 lowering 环节，但本例在 sm_86/11.5–13.0 仍复现。

---

## Root cause (preliminary, from archived trigger matrix)

cicc 13.3 前端错：13.3 的 PTX 在所有 ptxas 全部优化档下均错；归档矩阵中老版本的 O0 列也落在同一错值上（混合矩阵）。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Status: preliminary 归档标记为 unfixed；**深挖后确认本缺陷即已归档的 u16-signext 家族，CUDA 13.3 已修复**（13.3 全级别正确）。

## Minimized reproducer (`min_trigger.cu`, 2026-09-21 deep-dive)

基于 `test.cu` 全内核根因分析（见上方 confirmed 章节）做语句级消融后，
得到 **6 行标量**最小触发用例 `min_trigger.cu`（自判定 CORRECT/BUG）：

```cuda
__global__ void k(short* out, short x) {
    short v = (short)(-x);
    out[0] = (short)((v > (short)13569) ? 1 : 0);
}
// launch: k<<<1,1>>>(d, (short)-32768);
```

触发条件：`x = -32768`（INT16_MIN）。`-x` 数学上回绕仍为 `-32768`，
正确实现 `(v > 13569)` 为 **false**；错误实现以 32 位取负得到 `+32768`，
比较翻转为 **true**。与完整内核的连接：`test.cu` 中唯一触发线程
tid=1083（`input[1083] = 0x54b00004`，`(input>>5) & 0xFFFF = 0x8000`）
正是同一角落。

### 版本窗口（本机全工具链实测，sm_86）

| CUDA | `-O0/-O1/-O2/-O3` | `-G` |
| --- | --- | --- |
| 11.5 / 11.6 / 11.8 / 12.2 / 12.5 / 12.6 / 12.8 / 13.0 / 13.1 / 13.2 | `1`（**BUG**） | `0`（正确） |
| 13.3 | `0`（正确，**已修复**） | `0`（正确） |

（CUDA 11.1/11.3 工具链无法编译 sm_86，不可测。）

### SASS 铁证（12.6，-O3 错误 vs -G 正确）

错误（-O3）——取负后**缺失第二次 PRMT 再符号扩展**：

```
PRMT R0, R0, 0x9910, RZ        // sext16: 0x8000 → 0xFFFF8000 (-32768)
IMAD.MOV R0, RZ, RZ, -R0       // 32 位取负: 0x00008000 (+32768) ← 16 位截断被丢弃
ISETP.GT.AND P0, PT, R0, 0x3501 // +32768 > 13569 → TRUE（应为 FALSE）
```

正确（-G）——`PRMT …, 0x7610` + `PRMT …, 0x9910` 完成"16 位取负后再符号扩展"。

### 家族归并与状态更正

- 本缺陷与已归档的 `miscompile-O1plus-u16-signext0/1`
  （"ptxas omits the second PRMT re-sign-extension after neg.s16"，
  11.2..13.2，**fixed 13.3**）为**同一家族**：本最小复现即该家族的标量形态。
- 因此先期 "Status: unfixed as of CUDA 13.3" 不成立：**CUDA 13.3 已修复**
  （实测 13.3 全部 `-Xptxas` 级别与 -G 均为正确值 0）。
- 先期归因 "cicc 13.3 前端错" 不成立：`-Xptxas` 各级别 PTX 一致且语义正确，
  错误在 ptxas；11.5–13.2 错、13.3 对的版本窗口与 u16-signext 家族完全吻合。

## Files

| file | description |
| --- | --- |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums. |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C32
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
| CUDA 13.3 | -O0 | WRONG | 10138864006168268353 |
| CUDA 13.3 | -O1 | WRONG | 10138864006168268353 |
| CUDA 13.3 | -O2 | WRONG | 10138864006168268353 |
| CUDA 12.8 | -O0 | WRONG | 10138864006168268353 |
| CUDA 12.8 | -O1 | CORRECT | 433407214484750329 |

- source-correct checksum (GOOD side): `433407214484750329`
- miscompiled checksum (BAD side): `10138864006168268353`

## Provenance

- differential class: `C32` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C32_m0` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
