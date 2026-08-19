# 11.8_miscompile_1

自包含 CUDA-C 最小复现样例，从以下原始用例 reduce 得到：

`poc/cuda-11.8_miscompile/generator_ir__9e93ada826b9f09e/test.cu`

不使用 inline PTX，不依赖 `cuda_stress` Python 包或其他仓库代码；脚本只依赖本机已安装的 CUDA toolkit。

## 文件

- `runtime_minimal_no_inline.cu`（硬链接别名 `test.cu`）: 最小化 CUDA 源文件。
- `build_all.sh`: 用 CUDA 10.1 / 11.4 / 11.8，分别以 ptxas `-O0..-O3` 构建，并导出 SASS。
- `run_all.sh`: 默认先构建再运行 12 个二进制，打印结果矩阵。

默认工具链路径：

- `CUDA_10_1=/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- `CUDA_11_4=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- `CUDA_11_8=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`
- `GXX8=/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/bin/g++-8`

可通过环境变量覆盖，默认架构 `sm_75`，可用 `ARCH=...` 覆盖。

## 使用方法

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/11.8_miscompile_1
./run_all.sh
# 或先构建再运行
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

## 本机观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | OK checksum=32 | OK checksum=32 | OK checksum=32 | OK checksum=32 |
| cuda-11.4 | OK checksum=32 | OK checksum=32 | OK checksum=32 | OK checksum=32 |
| cuda-11.8 | OK checksum=0  | OK checksum=0  | OK checksum=0  | OK checksum=0  |

该样例完整保留了原始用例的差异形状：CUDA 10.1/11.4 输出一个值，CUDA 11.8 输出另一个值，且每个工具链在 ptxas 所有优化级别下稳定。

## Reduce 过程

原始完整用例（`test.cu` 原样编译运行）的 checksum 为：CUDA 10.1/11.4 得到
`14404487756982767718`，CUDA 11.8 得到 `2166945632205582236`。

1. 原始用例可先缩小为 `n=1`、`grid(1,1)`、`block(1,1)`，差异仍存在。
2. 删除未使用的 `helper_2` 以及 tiled-GEMM / bank-conflict / register-pressure 等后续代码，只保留 kernel 前奏；逐变量比较后，第一个分歧变量是 `v_28`。
3. `v_28` 只依赖 `n`、`v_19`、`v_25`；在原始 launch 下 `v_19 == 0`，`v_25 == (unsigned)(-1299614030)`，因此保留原始 `v_28` 表达式即可。
4. 关键触发条件：`n` 还必须有一个 32 位用途（原始用例中 `n` 被用于 `GPU_SMOD(tid, (unsigned)n)` 等）。若 `n` 只出现在 64 位左移中，NVVM 会直接按 64 位路径 lowering，10.1/11.4 与 11.8 的输出一致；保留最后的 `v_28 ^ (unsigned)n` 后差异出现。

## 根因（PTX 层）

`v_19=0` 时 `(~v_19) & 31 == 31`，`-(unsigned)(-1299614030) == 1299614030`，右移 31 位为 0。因此 `v_28` 等价于：

```cuda
__popcll((unsigned long long)n << 31u)
```

对 `n=1`，按 CUDA C++ 语义结果应为 `popcll(0x0000000080000000) == 1`，最终输出 `1 ^ 1 == 0`。

CUDA 10.1 / 11.4 生成的 PTX：

```ptx
ld.param.u32  %r1, [kernel_main_param_1];
mul.wide.s32  %rd3, %r1, -2147483648;
popc.b64      %r2, %rd3;
```

`mul.wide.s32 1, -2147483648` 得到符号扩展后的 `0xffffffff80000000`，popc 为 33，最终输出 `33 ^ 1 == 32`。

CUDA 11.8 生成的 PTX：

```ptx
ld.param.u32  %r1, [kernel_main_param_1];
cvt.s64.s32   %rd3, %r1;
shl.b64       %rd4, %rd3, 31;
popc.b64      %r2, %rd4;
```

`n=1` 时得到 `0x0000000080000000`，popc 为 1，最终输出 `1 ^ 1 == 0`。

PTX 在同一个工具链内不随 ptxas `-O0..-O3` 变化，因此这是 nvcc/NVVM 前端 lowering 差异，不是 ptxas 优化级别差异。

## 结论

该用例中，CUDA 11.8 的结果（checksum=0）符合源码语义；CUDA 10.1/11.4 把
`(unsigned long long)(int)n << 31` 错误 lowering 成带符号的 `mul.wide.s32 n, INT_MIN`。
原始流水线按 cross-build 多数投票把 CUDA 11.8 标成了 `buggy_build`，但这个具体
样例的投票方向与源码语义相反：11.8 修复了旧版本的前端 lowering 问题。
该最小样例仍可用于定位和复现原始跨版本结果不一致。
