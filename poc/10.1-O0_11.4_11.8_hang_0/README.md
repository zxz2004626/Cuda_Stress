# 10.1-O0_11.4_11.8_miscompile_0

自包含 CUDA-C 复现样例，从以下 prefix-scan differential-hang 用例 reduce 得到：

`poc/prefix_scan_multi_toolchain_hang/generator_ir__0ac10dd2c222f9fb/test.cu`

原始 923 行；当前自包含样例源码体 252 行（另加文件头注释）。

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`）：缩减后的 CUDA 源文件。
- `build_all.sh`：cuda-10.1 / 11.4 / 11.8 × ptxas `-O0..-O3` 构建并导出 PTX/SASS。
- `run_all.sh`：运行 12 个二进制；每个运行最多等待 4 秒，超时记为 `TIMEOUT`。

## 使用方法

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/10.1-O0_11.4_11.8_miscompile_0
./run_all.sh
```

## 观察结果（4 秒 timeout）

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | TIMEOUT | 14337055268274810065 | 14337055268274810065 | 14337055268274810065 |
| cuda-11.4 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |
| cuda-11.8 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |

与原始 pipeline 的 `hung_builds` 一致；10.1 O1+ 是仅有的完成构建。

## Reduce 过程（最终状态）

- launch：`n=8192, grid=(64), block=(128)` -> `n=32, grid=(1), block=(32)`
- `_rep`：3 -> 1
- 删除 register-pressure stress 块
- 删除三个未被调用的 helper 函数
- 删除 prefix scan 的全部 sweep 阶段；只保留 reduce 阶段
- 删除 `_ci` 后处理循环和 `_dead_0` 死代码块
- 用 statement-level delta debugging 删除可删除的前奏变量

尝试继续删除任意剩余语句或简化关键变量都会使至少一个目标 hang 消失；当前
252 行是保持完整 hang 矩阵的局部最小触发。

## 根因

- 每个工具链的 PTX 在所有 ptxas 优化级别下相同；差异来自 ptxas 生成的 SASS。
- 源码中的 `for` / `while` 都是有界循环，prefix-scan reduce 阶段的所有
  `__syncthreads()` 对 block 内线程是 uniform 的，不是源码层 divergent-barrier UB。
- 10.1 O1+ 生成可终止 SASS；10.1 O0 以及 11.4/11.8 全部优化级别生成的 SASS
  在相同 PTX 循环/前缀扫描结构上产生非终止调度。
- 删除嵌套 `w_0/i_0/w_1` 计算块可恢复 11.4/11.8 O1/O2，但 O0 仍超时；
  删除 `_fv` reduction 可恢复 11.4，但 11.8 仍超时。说明存在多个相互叠加的
  ptxas 调度错误，当前样例是同时覆盖全部目标 hang 的最小交集。
