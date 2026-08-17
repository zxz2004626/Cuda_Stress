# 10.1-O1_11.4-O2_miscompile_0

自包含 CUDA-C 复现样例，从以下 mixed-cross-toolchain 用例 reduce 得到：

`poc/mixed_cross_toolchain/generator_ir__ef89b4ca3680e9a2/test.cu`

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`）：缩减后的 CUDA 源文件。
- `build_all.sh`：cuda-10.1 / 11.4 / 11.8 × ptxas `-O0..-O3` 构建并导出 PTX/SASS。
- `run_all.sh`：运行 12 个二进制并打印结果矩阵。

默认路径与之前样例一致：

- `CUDA_10_1=/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- `CUDA_11_4=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- `CUDA_11_8=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`

## 使用方法

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/10.1-O1_11.4-O2_miscompile_0
./run_all.sh
# 或
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

## 观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 1 | **0** | 1 | 1 |
| cuda-11.4 | 1 | 1 | **0** | **0** |
| cuda-11.8 | 1 | 1 | 1 | 1 |

即原始 pipeline 的 buggy_builds：`cuda-10.1__O1` 和 `cuda-11.4__O2`；本样例显示
cuda-11.4 O3 与 O2 同侧。

## Reduce 过程

1. 原始 536 行用例在 `n=32768, grid=(128), block=(256)` 下复现目标差异。
2. 删除 register-pressure 与 switch-divergence stress 块，差异仍存在。
3. launch 缩到 `n=1, grid=(1), block=(32)`，差异仍存在；输出只差 1。
4. 将 `_rep` 循环固定为 1 次，差异仍存在，便于逐项裁剪。
5. 第一轮手工 reduce 得到 388 行 / 33.5 KB 的 `runtime_minimal.cu`。
6. 第二轮使用 `creduce` 加受限 interestingness test 继续压缩；曾得到 59 行 / 4.2 KB
   的版本，但该版本把输入初始化删掉了，会依赖未初始化 device memory，因此未采用。
7. 在强制保留 `cudaMemset(d_input, 0, ...)`、`cudaMemcpy`、`cudaFree`、无 inline asm
   的约束下继续压缩，并手工做单点/组合裁剪。最终源码为 51 行 / 2.1 KB。
8. 最终 `_fv` 只保留 3 个参与输出的项：

   ```cuda
   _fv = input[tid % n] ^ v_2 ^ v_5 ^ v_45;
   ```

   其中 `v_2 = (short)__byte_perm(tid, tid, 3u)`、`v_5 = gridDim.x`、`v_45`
   都不可直接替换为等值常量，否则 10.1 O1 或 11.4 O2 的差异会消失。`v_18`
   的 `(void)` 表达式、`arr_3[threadIdx.y][1]` global load、`v_30` 中的
   `__ldg(input[tid % n]) & v_6` 动态 shift 也必须保留；它们是让 ptxas 进入该
   错误调度/谓词化形状的上下文。

## 直接原因

最终输出只差 1：

```text
正确输出：1
错误输出：0
```

即 ptxas 误编译翻转/丢失了最终最低位贡献。当前最小版中输入显式置零，正确路径
等价于：

```cuda
output[0] = 0 ^ 0 ^ 1 ^ 0;  // 1
```

错误构建写出 `0`。因此问题出在 ptxas 对这段混合 `bool` / `unsigned int` /
`unsigned long long` 表达式和最终 XOR 的优化调度，而不是 host checksum、输入初始化
或数组越界。

替换实验：

- 把 `v_2` 或 `v_5` 替换成等值常量后，差异消失或只剩一侧工具链触发；
- 删除 `v_18` 这个看似 dead 的表达式，两个 buggy build 都恢复正常；
- 将 `arr_3[threadIdx.y][1]` 改成常量 0，或把 `v_30` 中两个动态 shift 都折叠为 0，
  差异消失；
- 将 `input[tid % n]` 相关 load 全部替换成 `input[0]`，11.4 O2 的差异消失。

## 根因

这是 ptxas 的优化误编译，不是源码 UB：

- 各工具链的 PTX 在所有 ptxas 优化级别下相同；不同输出来自 ptxas 生成的 SASS。
- 10.1 仅在 O1 触发，O0/O2/O3 均正确；11.4 在 O2/O3 触发，O0/O1 正确；
  11.8 全部正确。
- 触发形状是混合 `bool` / 整数表达式经由 `_fv = ... ^ v_2 ^ v_5 ^ v_45`
  归约到输出。ptxas 在重排/谓词化这些 XOR 时丢失或颠倒了一个
  与最低位相关的 predicate。
- 该样例中的除法/取模/移位均有 guard，device input 显式 `cudaMemset` 为 0，
  device globals 依赖 CUDA 零初始化，无未定义行为。

## Sanitizer

最终 51 行版本已用 CUDA 11.7 `compute-sanitizer` 检查 12 个构建：

- `memcheck`: 0 errors
- `racecheck`: 0 hazards
- `synccheck`: 0 errors
- `initcheck`: 0 errors

检查矩阵为 cuda-10.1 / 11.4 / 11.8 × ptxas `-O0..-O3`。
