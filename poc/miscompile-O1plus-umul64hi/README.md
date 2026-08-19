# 10.1-Op_11.4-Op_11.8-Op_miscompile_0

自包含 CUDA-C miscompile 复现样例，从以下 O0-vs-O1+ 差分用例 reduce 得到：

`poc/O0_vs_O1plus/generator_ir__29d6eb2c57873034/test.cu`

原始 `test.cu` 为 317 行 / 22100 bytes；当前最小样例为 21 行 / 716 bytes。

## 文件

- `test.cu`：缩减后的 CUDA 源文件。
- `runtime_minimal.cu`：与 `test.cu` 内容相同（硬链接）；`build_all.sh` 默认编译该文件。
- `build_all.sh`：构建 cuda-10.1 / 11.4 / 11.8 × ptxas `-O0..-O3`，并导出 PTX/SASS。
- `run_all.sh`：运行 12 个二进制并打印结果矩阵。

默认路径与之前样例一致：

- `CUDA_10_1=/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- `CUDA_11_4=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- `CUDA_11_8=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`

## 使用方法

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/10.1-Op_11.4-Op_11.8-Op_miscompile_0
./run_all.sh
# 或
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

## 观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 0 | **1027807174** | **1027807174** | **1027807174** |
| cuda-11.4 | 0 | **1027807174** | **1027807174** | **1027807174** |
| cuda-11.8 | 0 | **1027807174** | **1027807174** | **1027807174** |

原始 pipeline 的 cross-build consensus 把三工具链的 O0 一起标记为 `buggy_builds`
（O0 组输出与 O1/O2 组不同）。本最小样例可以直接确定 ground truth：

```cuda
unsigned int v_4 = 4042322160u;                 // kernel 参数，运行时传入
unsigned int v_6 = (unsigned int)__umul64hi(
    1027807175u,
    (unsigned long long)__umulhi(gridDim.x, v_4) - threadIdx.y);
```

对于 `<<<1,1>>>` 启动，`gridDim.x == 1`、`threadIdx.y == 0`，因此：

```text
__umulhi(1, 4042322160) == 0
0 - 0 == 0
__umul64hi(1027807175, 0) == 0
```

所以输出 `0` 才是正确值；O1/O2/O3 三套工具链都输出 `1027807174`
（`0x3d41b286`），属于 ptxas miscompile。

## Reduce 过程

1. 原始 317 行 / 22100 bytes 用例在 `n=32768, grid=(128), block=(256)` 下复现目标差异。
2. 仅缩小 launch 到 `n=1, grid=(1), block=(1)`，差异仍稳定存在。
3. 删除未调用的 `helper_1`、恒假分支中的 `helper_0`、未使用的 `ov_1..ov_7`
   以及大量 register-pressure / switch-divergence / 内层 stress 循环，矩阵保持不变。
4. 对剩余 kernel 做数据流裁剪：`_fv` 最终 XOR 链中只有 `v_6` 的初始表达式不可折叠；
   把 `v_6` 替换为常量 `0` 后 6 个构建全部输出 `0`，差异消失，从而定位到
   `__umul64hi` / `__umulhi` 表达式本身。
5. 进一步把该表达式单独放进一个最小 kernel，cuda-10.1 / 11.4 / 11.8 的
   ptxas `-O0` 与 `-O1/-O2/-O3` 仍完整复现原矩阵。
6. 单点替换实验：

   - 把 `v_4` 直接写成常量 `4042322160u` 会触发前端常量折叠，差异消失；
     必须让 `v_4` 是运行时值（本样例用 kernel 参数）。
   - 去掉 `- threadIdx.y` 后 O1+ 恢复正常；64 位减法与 `threadIdx.y`
     的耦合是触发条件。
   - 把 `__umulhi` 结果和 64 位减法拆成独立局部变量后 O1+ 也恢复正常；
     必须保留“嵌套在 `__umul64hi` 第二操作数中的 64 位 zero-minus-tid.y”形状。
   - 第一操作数写成 `1027807175u` 或 `1027807175ull` 均可，外层 `(unsigned int)`
     转换只影响 PTX 的截断表示，不影响触发矩阵。

最终源码为 21 行 / 716 bytes。

## 直接原因

各工具链的 PTX 在所有 ptxas 优化级别下语义相同，核心片段为：

```ptx
mul.hi.u32 %r3, %nctaid.x, %r1;   // = 0（1 * 4042322160 的高 32 位）
cvt.u64.u32 %rd3, %r3;            // = 0
mov.u32 %r4, %tid.y;
cvt.u64.u32 %rd4, %r4;            // = 0
sub.s64 %rd5, %rd3, %rd4;         // 语义应为 0
mul.hi.u64 %rd7, 1027807175, %rd5;
```

cuda-10.1 的 SASS 对比显示：

- `-Xptxas -O0`：按 64 位减法正确展开，`sub.s64` 的高 32 位保持 `0`。
- `-Xptxas -O1`：把 `sub.s64 0, 0` 的扩展字错误固化为 `-1`：

  ```sass
  IADD3   R0, RZ, -R0, RZ        ; low = 0 - tid.y
  IADD3.X R7, RZ, -0x1, RZ, ...  ; high 被置为 0xffffffff
  IMAD.WIDE.U32 R2, R7, 0x3d4317c7, RZ
  ```

  后续 `mul.hi.u64` 使用 `0xffffffff_00000000`（或等价的错误高位扩展）作为
  第二操作数，得到 `floor(1027807175 * (2^64 - 1) / 2^64) = 1027807174`。

cuda-11.4 / 11.8 的 O1+ SASS 存在相同的错误高位扩展形状，因此三套工具链在
O1/O2/O3 一起 miscompile；O0 均正确。

## 根因判断

- 源码无 UB：单线程、无越界、无未初始化内存、无 barrier/race；
  `__umulhi` 和 `__umul64hi` 都是无符号整型 intrinsic，结果完全确定。
- `-Xptxas -O*` 不改变前端 PTX；差异只来自 ptxas 生成的 SASS。
- ptxas 在优化 `cvt + sub.s64 + mul.hi.u64` 这组零值/特殊寄存器依赖时，
  错误地把 64 位减法结果的高位扩展成 `-1`，导致 `mul.hi.u64` 使用错误的
  64 位操作数。
- 原始 `O0_vs_O1plus` 只是按输出分组把它标成 “O0 偏离 O1/O2”；最小化后可以
  证明 O0 是正确侧，O1/O2（以及 O3）才是 miscompile 侧。

## Sanitizer

最终 21 行版本已用 CUDA 11.7 `compute-sanitizer` 检查 12 个构建：

- `memcheck`: 0 errors
- `racecheck`: 0 hazards
- `synccheck`: 0 errors
- `initcheck`: 0 errors

检查矩阵为 cuda-10.1 / 11.4 / 11.8 × ptxas `-O0..-O3`，完整日志见
`build/sanitizer.log`。
