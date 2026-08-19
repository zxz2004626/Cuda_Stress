# 10.1-O0_11.4_11.8_hang_0

自包含 CUDA-C hang 复现样例，从以下 prefix-scan multi-toolchain hang 用例继续 reduce 得到：

`poc/prefix_scan_multi_toolchain_hang/generator_ir__0ac10dd2c222f9fb/test.cu`

原始 923 行；当前样例为 49 行 / 4481 bytes，不使用 inline PTX。

## 文件

- `test.cu`：缩减后的 CUDA 源文件。
- `runtime_minimal.cu`：与 `test.cu` 内容相同；`build_all.sh` 默认编译该文件。
- `build_all.sh`：构建 cuda-10.1 / 11.4 / 11.8 × ptxas `-O0..-O3`，并导出 PTX/SASS。
- `run_all.sh`：运行 12 个二进制；每个运行最多等待 4 秒，超时记为 `TIMEOUT`。脚本会先运行预期能结束的 cuda-10.1 O1/O2/O3，避免先前 timeout 污染正常项判断。

## 使用方法

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/10.1-O0_11.4_11.8_hang_0
./run_all.sh
```

## 观察结果（4 秒 timeout）

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | TIMEOUT | OK | OK | OK |
| cuda-11.4 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |
| cuda-11.8 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |

与原始 pipeline 的 hang 矩阵一致：cuda-10.1 只有 O1/O2/O3 可以结束；cuda-10.1 O0 和 cuda-11.4/11.8 全部 ptxas 优化级别都会非终止。

## Reduce 过程（当前状态）

- launch 从大规模 prefix-scan case 缩到 `kernel_main<<<1, 32>>>(..., n=32)`。
- 删除 host 侧输入初始化和结果校验，只保留 `cudaMalloc`、`cudaMemset`、kernel launch、`cudaDeviceSynchronize`。
- 删除 prefix-scan sweep、后处理循环、register-pressure 块、未调用 helper 和大量前奏变量。
- 删除 `#include <cuda_runtime.h>`、`v34`、`v42`、`v14`、`v16` 等对当前触发条件不必要的代码。
- 将 `v4` 常量化为 0，简化 `v11` 的全局数组访问，缩短部分等价类型和 `tid` 表达式。

以下看似可继续删除/化简的部分仍会破坏原始矩阵，因此保留：

- `extern "C"` kernel/global 符号形态。
- `arr_1` 声明，即使当前 `v4=0` 后源码层不再读取它。
- `arr_2` 全局数组 load、`v6`、`v17`、`v28`、`v30`、`v33`、`v35`、`v47`。
- uniform `__syncthreads()` 和嵌套 `w_0/i_0/w_1` 有界循环。
- `GPU_SDIV/GPU_SMOD/GPU_SSHL/GPU_SSHR` 的 guard 形态；直接改成裸除法、裸取模或裸 shift 会改变触发矩阵。

## 根因判断

- 源码层没有发现导致 hang 的 CUDA UB：所有循环有界；唯一的 `__syncthreads()` 对 block 内所有线程 uniform；`__any_sync(~0u, tid)` 由全部 active lanes 以同一 mask 调用；数组索引、除法、取模和 shift 均有 guard。
- `-Xptxas -O*` 不改变前端 PTX；同一 toolchain 内的差异来自 ptxas 生成的 SASS。
- 10.1 O1/O2/O3 生成可终止 SASS；10.1 O0 以及 11.4/11.8 全部优化级别在同一个小型有界循环/全局 load/warp vote/barrier 组合上生成非终止执行路径。
- 因此该 case 的核心不是源码死循环，而是 ptxas 后端在特定控制流和寄存器依赖形态下的 SASS 控制流/调度错误；11.4/11.8 的触发范围比 10.1 更宽。
