# C33: cicc dual-column miscompile: 11.1 and 13.3 both emit wrong PTX, 11.3..13.0 correct

Self-contained, stable, UB-free reduced reproducer distilled from the
Cuda_Stress differential run (trigger-matrix class **C33**).

## Trigger matrix (archived, sm_86, CUDA 11.1..13.3 × ptxas O0/O1/O2)

`11.1/O0-O1-O2 + 13.3/O0-O1-O2（双列错：11.1 整列 + 13.3 整列，中间版本全对）`

The archived matrix groups the 27 (version × ptxas -O) builds by output
checksum; the listed cells are the minority (deviating) builds.

## Root cause (confirmed, deep-dive analysis of `test.cu`)

**结论：源码无 UB。NVVM 中端两处错误叠加，且均为优化级别依赖
（-O0..-O3 错、-G 对）：**

1. `(unsigned long long)__mul24(x,x) % 大常数` 的取模被整体删除
   （把 `__mul24` 的 int 截断结果按"精确乘积"建模，等效于零扩展）——
   与 C22 同族；
2. `^ helper` 被写成 `| helper`（OR 与 XOR 仅在操作数与常数无公共位时等价，
   实际 `helper = 2` 与常数 0xFFFFFFF2 的 bit1 相撞）。

### 实测（RTX 3090, sm_86，重测确认）

- 标准/源语义（Python 模拟 = g++ 主机参考 = nvcc `-G`，11.5 与 13.0）：`10261298828832264655`
- nvcc 11.5/12.6/13.0 的 `-O0/-O1/-O2/-O3`：`291630905467910284`（每个输出 = 正确值 ^ 0x3B，
  即 `v_38` 项差：helper 由 2 变 1）

### PTX / SASS 证据

- -G PTX 正确：`cvt.s64.s32`（符号扩展）+ `rem.u64`（真取模）；
- -O0/-O3 PTX（两者一致）错误：`mul.lo.s32` 后取模消失，
  `or.b32 %r15, %r14, -14` 应为 `xor.b32`；
- 独立小内核中同一表达式保留符号扩展与取模 → 错误依赖上下文；
- 显式 `(unsigned long long)(long long)__mul24(...)` 无法阻止（模型贯穿后续转换）。

---

## Root cause (preliminary, from archived trigger matrix)

cicc 前端（双列形态）：13.3 与 11.1 的 PTX 在全部 ptxas 下都错，11.3–13.0 正确。双版本列矩阵在归档 poc 中无先例。

Evidence (cross-ptxas sweep, see `rc_sweep` reference below): the failing-side
PTX was assembled with every toolkit's ptxas and executed; the per-version /
per-opt checksum matrix pins the defect to the compiler layer and window shown
above. The reduced core is deterministic across repeated runs and clean under
compute-sanitizer memcheck / initcheck / racecheck.


Status: preliminary 归档（基于 reduced core）标记为 unfixed；**深挖确认 `test.cu` 实例在 CUDA 13.3 前端已输出正确结果**（13.3 的 -O0/-O3 与 -G 一致给出 0xffffcfc9；11.5–13.2 的 -O0/-O3 仍错）。

## Minimized reproducer (`min_trigger.cu`, 2026-09-21 deep-dive)

基于 `test.cu` 全内核根因分析做语句级消融后，得到的最小触发用例
（helper 为 `__device__` 函数是必要条件——内联进 kernel 后不触发）：

```cuda
#define DEN 1431655765u
__device__ unsigned h(void) {
    unsigned p1 = 0x55555555u;                            // = ~(unsigned)__brev(1431655765u)
    int v = (int)__byte_perm(p1, p1, 4u) * 7;             // = 0x55555553 (1431655763)
    int cse = ((int)(((unsigned)v & 16777215u) ^ 8388608u)) - 8388608;  // = 5592403
    return 1u * ((unsigned)__mul24(cse, cse) / DEN);      // = 2 (0xC58E38E9 / 0x55555555)
}
__global__ void kernel_main(unsigned int* out, const unsigned int* in) {
    unsigned fv = in[0] ^ 0xAAAAAAAAu ^ (0x55555558u ^ h());
    out[0] = fv;   // 正确: in ^ 0xFFFFFFF0; nvcc: in ^ 0xFFFFFFF2
}
```

### 版本窗口（本机全工具链实测，sm_86，-O0/-O3）

| CUDA | 结果 | 判定 |
| --- | --- | --- |
| 11.5 / 11.6 / 11.8 / 12.2 / 12.5 / 12.6 / 12.8 / 13.0 / 13.1 / 13.2 | `0xffffcfcb` | **BUG** |
| 13.3（-O0/-O3） | `0xffffcfc9` | CORRECT |
| 全版本 `-G` | `0xffffcfc9` | CORRECT |

### 机制

- helper 的运行时值 = 2（`__mul24(5592403,5592403) = 0xC58E38E9`，
  `0xC58E38E9 / 0x55555555 = 2`，`__ffsll(1) = 1`）；
- NVVM 将 `out = in ^ 0xAAAAAAAA ^ (0x55555558 ^ h)` 中的 `^ h`
  替换为 `| h`（OR≡XOR 需 h 的位 ⊆ 常数 0xFFFFFFF2 的零位 {0,2,24..31}；
  实际 h = 2 撞 bit1）→ 输出常数 0xFFFFFFF2 而非 0xFFFFFFF0；
- 同族：C22（`(ull)__mul24 % C` 取模删除）为同一"精确乘积建模"的另一表现。

## Files

| file | description |
| --- | --- |
| `test.cu` | Self-contained minimal reproducer (device kernel + host `main` that prints `checksum=<u64>`). |
| `Makefile` | Builds the BAD and GOOD configurations and checks both checksums. |
| `metadata.yaml` | Machine-readable summary (trigger, checksums, provenance). |

## Usage

```bash
cd /export/ssd1/xzhanglb/llm-template/Cuda_Stress/poc/C33
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
| CUDA 13.3 | -O0 | WRONG | 10261298828832264655 |
| CUDA 13.3 | -O1 | WRONG | 10261298828832264655 |
| CUDA 13.0 | -O0 | CORRECT | 291630905467910284 |
| CUDA 13.0 | -O1 | CORRECT | 291630905467910284 |

- source-correct checksum (GOOD side): `291630905467910284`
- miscompiled checksum (BAD side): `10261298828832264655`

## Provenance

- differential class: `C33` of run set
  Cuda_Stress(2026-09-11) / Cuda_Stress_ds(2026-09-14) / Cuda_Stress_glm(2026-09-13) /
  Cuda_Stress_glmmax(2026-09-15) / Cuda_Stress_gpt(2026-09-16)
- reduced core: `reduce_out/C33_m0` (`textreduced.cu`), produced by
  `generator_ir.reduce` with UB gates; verification: `ub_check.json`, cross-ptxas
  evidence: `rc_sweep.json` in the same directory.
- 27-cell faithful re-sweep of the class representative: `trigger_matrix_analysis/faithful_sweep.jsonl`.
