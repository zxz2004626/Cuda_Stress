# 11.4_miscompile_2

自包含 CUDA-C 复现样例，从以下用例 reduce 得到：

`poc/cuda-11.4_miscompile/generator_ir__9f23fc3cb34e3ff8/test.cu`

不使用 inline PTX；所有移位均有 guard 或合法 shift amount，无源码 UB。

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`，当前 23 行 / 980 字节）
- `build_all.sh`
- `run_all.sh`

## 观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 197 | 197 | 197 | 197 |
| cuda-11.4 | 197 | **229** | 197 | 197 |
| cuda-11.8 | 197 | 197 | 197 | 197 |

该样例只在 cuda-11.4 `ptxas -O1` 出错，O0/O2/O3 均正确。

## Reduce 结果

- 原始 606 行 / 45657 字节，当前 23 行 / 980 字节。
- 输出定位到原 kernel 第 256 号输出（block 1 的 thread 0），差分插桩显示
  该输出在 `_fv ^= (unsigned)v_39` 处开始分歧，继续上溯到 `v_49`。
- 去掉原 kernel 中与 `v_49` 数据流无关的循环、shared memory 和 helper 后，
  最终核心为：

  ```cuda
  #define GPU_SSHR(a, b) ((a) >> (b))

  __global__ void kernel_main(unsigned int* output, signed char var_2) {
      unsigned int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
      long long v_11 = (37u ^ GPU_SSHR(tid, (tid & 31u)));
      unsigned int v_16 = GPU_SSHR(
          ((0 < threadIdx.x) ? ((var_2 > (signed char)blockDim.x) ? v_11 : 0) : 0), 0);
      signed char v_24 =
          ((0 != (unsigned char)__byte_perm(1431655765u, threadIdx.x, 2u)) ? 0 : 49);
      unsigned char v_49 =
          ((v_16 ^ GPU_SSHR(2739585995u, (blockDim.y & 31u))) ^
           GPU_SSHR(__funnelshift_rc(0, (blockDim.y ^ v_24), 15u), 12u));
      if (tid == 256u) output[0] = v_49;
  }
  ```

## 根因

### PTX 层

三个工具链生成的 PTX 语义一致，关键片段为：

```ptx
prmt.b32 %r20, %r19, %r2, 2;      // __byte_perm(1431655765, threadIdx.x, 2)
and.b32  %r21, %r20, 255;
setp.eq.s32 %p3, %r21, 0;         // 正确路径应使用这个谓词
selp.b32 %r22, 49, 0, %p3;        // v_24 = (byte_perm == 0) ? 49 : 0
```

### SASS 层

cuda-11.4 `ptxas -O1` 生成的 SASS 中：

```sass
ISETP.NE.AND P0, PT, R5, RZ, PT ;   // P0 = (threadIdx.x != 0)
PRMT RZ, R3, 0x7772, RZ ;           // byte_perm 结果被写回 RZ，等于丢弃
SEL  R3, RZ, 0x31, P0 ;             // 错误复用 P0，而非 byte_perm 谓词
```

本样例观察的是 `tid == 256`，所以 `P0 = (threadIdx.x != 0)` 为 true，
`v_24` 被错误置为 49；正确值应为 0。

对照 cuda-11.4 O2 / cuda-10.1 / cuda-11.8：

```sass
PRMT RZ, R4, 0x7772, RZ
SEL  R4, RZ, 0x31, P1               // P1 来自 var_2/blockDim 路径，结果正确
```

`v_24` 进入 `v_49` 的 `__funnelshift_rc(0, blockDim.y ^ v_24, 15)`，
因此错误值 49 最终使 `v_49` 从 197 变成 229。

### 结论

与 `11.4_miscompile_1` 类似的 ptxas -O1 谓词复用错误：cuda-11.4 O1 在优化
`__byte_perm(...) != 0 ? ... : ...` 时把 PRMT 结果写回 `RZ`，并删除了
`and + setp.eq` 这组谓词重算指令，导致 `SEL` 复用了前面 `0 < threadIdx.x`
guard 遗留的谓词 P0。O0 未做该优化，O2 恰好使用了正确的另一个谓词 P1，
因此错误只在 O1 出现。
