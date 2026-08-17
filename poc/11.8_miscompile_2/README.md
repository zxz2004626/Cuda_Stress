# 11.8_miscompile_2

自包含 CUDA-C 最小复现样例，从以下原始用例 reduce 得到：

`poc/cuda-11.8_miscompile/generator_ir__35d576a5d6f55918/test.cu`

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
cd /export/d3/xzhanglb/Cuda_Stress/poc/11.8_miscompile_2
./run_all.sh
# 或先构建再运行
./build_all.sh
SKIP_BUILD=1 ./run_all.sh
```

## 本机观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | checksum=1 | checksum=1 | checksum=1 | checksum=1 |
| cuda-11.4 | checksum=1 | checksum=1 | checksum=1 | checksum=1 |
| cuda-11.8 | checksum=1 | checksum=0 | checksum=0 | checksum=0 |

原完整用例的差异形状为：cuda-10.1/11.4 与 cuda-11.8 O0 输出相同，cuda-11.8 O1/O2
输出不同（原始 `summary.json` 中 O3 未参与执行，本样例显示 O3 与 O1/O2 同侧）。

## 这个 kernel 在 n=1, grid=(1), block=(32) 时计算什么

对 `tid == 0`：

```cuda
v_0  = gridDim.x = 1;
v_1  = (bool)(208952228u ^ ((unsigned)(GPU_SMOD(0, 2654435769u)) * (unsigned)0));
     = (bool)(208952228u ^ 0) = true;
v_18 = (bool)(... * var_0) = (bool)(... * false) = false;

// GPU_SSHR 的 guard 使负移位量返回 0，所以：
v_8  = ~GPU_SSHR(167, GPU_SDIV(-7293850184320197ll, 1));
     = ~0 = -1;
(unsigned char)v_8 = 255;

v_36 = GPU_SMOD(v_1 - v_18, (unsigned char)v_8)
     = GPU_SMOD(1, 255) = 1;          // bool true

output[0] = (unsigned)v_36 = 1;
```

因此源码语义结果为 **1**。cuda-11.8 的 ptxas `-O1..-O3` 把 `v_36` 算成了
**0**，而 cuda-11.8 `-O0` 以及 cuda-10.1/11.4 的所有优化级别都得到 **1**。

代码中的除法/取模/移位宏都有 guard，无符号运算按模 2^N 定义；`__ldg`、atomic
和无符号转换均合法。该样例不存在由源码 UB 导致差异的问题。

## Reduce 过程

1. 原始用例可编译运行复现：cuda-10.1/11.4 与 cuda-11.8 O0 输出一个 checksum，
   cuda-11.8 O1/O2 输出另一个 checksum。
2. 把原始 32768 输出逐一与 11.8 O0 比较，发现所有输出只差 `+/-1`；进一步在
   kernel 末尾把 61 个中间变量导出到 output 后比较，发现只有 `v_36` 不同
   （O0=1，O1=0），最终输出差恰好来自 `_fv ^ (unsigned)v_36`。
3. 删除未使用的 `helper_0`、`helper_1`、`__device__` 数组、bank-conflict /
   register-pressure / vectorized-load / switch-divergence stress 块，差异仍存在。
4. 把 launch 缩小到 `grid=(1), block=(32), n=1`，差异仍存在。
5. 逐变量简化/删除 kernel 前奏中的死变量。大部分变量被简化成零或删除；剩余变量
   （尤其 `v_9/v_10/v_12/v_16/v_17/v_23/v_28/v_31/v_45` 以及 atomicAdd 循环）
   是保留触发条件所必需的上下文。它们单独删除或简化都会让 ptxas 的寄存器调度发生
   变化，导致 O1 误编译消失，因此这些“看起来无用”的语句被刻意保留。

最终源文件约 100 行，跨版本差异形状完整保留。

## 根因（PTX/SASS 层）

nvcc 11.8 为所有 ptxas 优化级别生成**相同**的 PTX，因此问题出在 ptxas 11.8 的
`-O1` 及更高优化，而不是 nvcc/NVVM 前端 lowering。

核心 PTX（cuda-11.4 与 cuda-11.8 在这一段基本一致）：

```ptx
not.b64   %rd11, %rd83;          // v_8 = ~(shift result)
...
and.b64   %rd67, %rd11, 255;     // (unsigned char)v_8 的 64 位掩码
setp.eq.s64 %p23, %rd67, 0;      // 除零 guard
mov.u32   %r54, 0;               // v_36 的默认值
@%p23 bra $L__BB0_31;            // 若分母为 0 则跳过余数计算
...
cvt.u32.u64 %r40, %rd11;
and.b32   %r41, %r40, 255;       // 分母 = 255
rem.s32   %r42, %r39, %r41;      // v_36 = 1 % 255
setp.ne.s32 %p27, %r42, 0;
selp.u32  %r54, 1, 0, %p27;
$L__BB0_31:
...
st.global.u32 [%rd79], %r54;
```

本样例中 `not` 之前的值 `%rd83 == 0`，所以 `not` 之后 `%rd11 == -1`，分母应为
`255`，guard 不应跳转。ptxas 11.4 O1 的 SASS 用两条 predicate LOP3 组合 64 位
零检查后再进入 `rem`；ptxas 11.8 O1 的 SASS 把该 guard 折叠成单条 predicate
LOP3，并在 `not` 之前的低 32 位为 0 时直接跳过了余数计算，使 `v_36` 停留在初始
值 0，最终输出 0。

11.8 O1 的关键 SASS（省略无关部分）：

```sass
SHF.R.U64 R4, R3, R4, RZ ;           // R4 = not 之前的 shift result
LOP3.LUT P1, RZ, R4, 0xff, RZ, 0xc0 ; // 11.8: 单条 guard
MOV R2, RZ ;                          // v_36 = 0
@!P1 BRA `(.L_x_4) ;                  // 本样例中错误跳过余数计算
LOP3.LUT R7, R4, 0xff, RZ, 0xc ;      // R7 = (~R4) & 0xff = 255
...
SEL R2, RZ, 0x1, !P1 ;
```

而 cuda-11.4 O1 在同处是：

```sass
LOP3.LUT P0, RZ, R4, 0xff, RZ, 0xc, !PT ;
LOP3.LUT P0, RZ, R15, RZ, RZ, 0xc, P0 ;
...
@!P0 BRA `(.L_5) ;
LOP3.LUT R7, R4, 0xff, RZ, 0xc, !PT ;
```

为验证“guard 看的是 `not` 之前的值”，把源码中 `GPU_SDIV(-7293850184320197ll, v_0)`
临时改成 `GPU_SDIV(0ll, v_0)`，使 `not` 前的低 32 位为 167（非 0），11.8 O1
即恢复输出 1。差异消失，与上述判断一致。

## 结论

- 这是 **CUDA 11.8 ptxas `-O1` 及以上优化级别**的误编译，具体触发点是带 64 位
  `and + setp.eq` 零分母 guard 的 `rem.s32` 序列。
- 11.8 O1 在把 `not.b64` 与 guard 折叠时使用了 `not` 之前的低 32 位值做零检查，
  本例中错误跳过了 `v_36 = 1 % 255`，保留了默认值 0。
- cuda-10.1/11.4 的所有优化级别以及 cuda-11.8 O0 都输出源码语义值 1。
- 本最小样例可稳定复现：11.8 的 `-Xptxas -O1/-O2/-O3` 输出 0，其余输出 1。
