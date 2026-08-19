# 11.8_miscompile_3

自包含 CUDA-C 最小复现样例，从以下原始用例 reduce 得到：

`poc/cuda-11.8_miscompile/generator_ir__38c3f9cb2f485ed7/test.cu`

本样例不使用 inline PTX，只依赖 CUDA toolkit、gcc-8 和一张可用 GPU。

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`）：最小化 CUDA 源文件。
- `build_all.sh`：用 CUDA 10.1 / 11.4 / 11.8，分别以 ptxas `-O0..-O3` 构建，并导出 PTX 与 SASS。
- `run_all.sh`：默认先构建再运行 12 个二进制，打印结果矩阵。

默认工具链路径：

- `CUDA_10_1=/export/d3/xzhanglb/nvcc-toolkit/cuda-10-1`
- `CUDA_11_4=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-4`
- `CUDA_11_8=/export/d3/xzhanglb/nvcc-toolkit/cuda-11-8`
- `GXX8=/export/d3/xzhanglb/nvcc-toolkit/gcc-8/usr/bin/g++-8`

默认架构 `sm_75`，可用 `ARCH=...` 覆盖。

## 使用方法

```bash
cd /export/d3/xzhanglb/Cuda_Stress/poc/11.8_miscompile_3
./run_all.sh
# 或先构建再运行
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

## 本机观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 65520 | 65520 | 65520 | 65520 |
| cuda-11.4 | 65520 | 65520 | 65520 | 65520 |
| cuda-11.8 | 65520 | 3 | 3 | 3 |

## 最小样例计算什么

kernel 参数 `x = 0`，`blockIdx.x = 0`：

```cuda
output[0] =
  (unsigned)(unsigned short)(signed char)(
      (~(unsigned long long)x) /
      ((unsigned long long)__clzll((unsigned long long)blockIdx.x) | 1ULL));
```

代入：

```text
~0ULL = 0xffffffffffffffff
__clzll(0) = 64
denominator = 64 | 1 = 65

0xffffffffffffffff / 65 = 0x3f03f03f03f03f0
low byte = 0xf0 -> signed char -16 -> unsigned short 0xfff0 = 65520
```

所以源码语义结果为 **65520**。CUDA 11.8 的 ptxas `-O1/-O2/-O3` 得到 **3**。

`3` 正好是 32 位除法路径的结果：

```text
~0U = 0xffffffff
0xffffffff / 65 = 0x03f03f03
low byte = 0x03
```

代码中没有 UB，也没有未定义移位/除法。

## Reduce 过程

1. 原始 904 行用例复现：cuda-10.1/11.4 与 cuda-11.8 O0 同输出，cuda-11.8 O1/O2
   不同。
2. 删除 register-pressure / vectorized-load / switch-divergence stress 块，差异仍存在。
3. 把 launch 缩到 `n=1, grid=(1), block=(32)`，差异仍存在；进一步 block=(1) 也复现。
4. 在 kernel 末尾导出 59 个中间变量比较 O0/O1，发现唯一直接分歧变量是 `v_38`
   （O0=65520，O1=3），其余差异均由它传播。
5. 定位到 `_rep` 循环 else 分支里的：
   `v_38 = (signed char)(~arr_2[...] / (clz(...) | 1))`。
   强制走该分支后差异仍存在；删除该表达式则差异消失。
6. 逐步删除 input、tid、arr_2 二维数组、循环等上下文，最终得到上面的单表达式
   kernel。分子必须来自动态值（kernel 参数即可），写成常量 `~0ULL` 不会触发；
   分母也必须由 `__clzll(blockIdx.x)` 动态生成，写成常量 65 不会触发。

## 根因（PTX/SASS 层）

nvcc 11.8 为所有 ptxas 优化级别生成相同 PTX。核心 PTX 是 NVVM 对 64 位除法的
“先检查高 32 位，能用 32 位除法就用 32 位”的常规 lowering：

```ptx
ld.param.s32 %rd7, [kernel_main_param_1];  // x = 0
not.b64      %rd1, %rd7;                   // numerator = ~x = 0xffffffffffffffff
...
and.b64      %rd10, %rd1, -4294967296;     // 检查 numerator 高 32 位
setp.eq.s64  %p1, %rd10, 0;
@%p1 bra     $L__BB0_2;                    // 高 32 位为 0 才走 32 位除法

div.u64      %rd12, %rd1, %rd2;            // 64 位路径：正确
bra.uni      $L__BB0_3;

$L__BB0_2:
cvt.u32.u64  %r3, %rd2;
cvt.u32.u64  %r4, %rd1;
div.u32      %r5, %r4, %r3;                // 32 位快速路径
cvt.u64.u32  %rd12, %r5;

$L__BB0_3:
...
st.global.u32 [%rd11], %r6;
```

本例 `~x` 的高 32 位是 `0xffffffff`，所以 **不应** 走 `div.u32`，而应走
`div.u64`。ptxas 11.8 O1 的 SASS 错误地使用了 `not` 之前的值来生成这个 guard：

```sass
USHF.R.S32.HI UR4, URZ, 0x1f, UR5 ;  // UR4 = x 的符号扩展高 32 位，x=0 -> 0
ULOP3.LUT UR5, URZ, UR5, URZ, 0x33 ; // 低 32 位先算 ~x
ISETP.NE.U32.AND P0, PT, RZ, UR4, PT ; // guard 用的是 pre-not 高 32 位！
ULOP3.LUT UR4, URZ, UR4, URZ, 0x33 ; // 高 32 位的 ~x 在 guard 之后才算
@!P0 BRA 0xe0 ;                      // x=0 时错误分支到 32 位除法路径
I2F.U32.RP R4, R0 ;                  // 0xffffffff / 65 -> 低字节 0x03
```

也就是说：

```text
正确 guard：high32(~x) != 0  -> 走 div.u64
11.8 O1：   high32( x) != 0  -> 走 div.u64
```

当 `x >= 0` 时 `high32(x) == 0`，ptxas 11.8 O1 误选 `div.u32`；当 `x < 0`
（例如 -1）时 guard 恰好又“碰对”，O0/O1 输出一致。验证结果：

- `x = 0` 或 `x = 1`：O0=65520，O1/O2/O3=3（错误）
- `x = -1`：O0=0，O1/O2/O3=0（一致）

cuda-11.4 O1 在同处先正确生成 `~x` 的高 32 位再测试 guard，因此始终走 64 位
路径，输出 65520。

## 结论

- 这是 **CUDA 11.8 ptxas `-O1` 及以上优化级别**的误编译。
- 触发模式是 NVVM 生成的 64 位除法窄化 guard：

  ```ptx
  not.b64 %rd, %rsrc;
  and.b64 %rhigh, %rd, -4294967296;
  setp.eq.s64 %p, %rhigh, 0;
  @%p bra 32-bit-path;
  ```

- ptxas 11.8 O1 在 predicate 生成时把 `not.b64` 放到了 guard 之后（或等价地，
  用 `not` 前的高 32 位做判断），导致高 32 位非零的 `~x` 错误走进 `div.u32`。
- 结果从源码语义值 65520 变成 32 位除法结果 3。
