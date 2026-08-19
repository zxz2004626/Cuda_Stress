# 11.4_miscompile_1

自包含 CUDA-C 复现样例，从以下用例 reduce 得到：

`poc/cuda-11.4_miscompile/generator_ir__7c080bf20e32469b/test.cu`

不使用 inline PTX；取模带 guard，无源码 UB。

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`，当前 25 行 / 915 字节）
- `build_all.sh`
- `run_all.sh`

## 观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 2654448107 | 2654448107 | 2654448107 | 2654448107 |
| cuda-11.4 | 2654448107 | **2654448106** | **2654448106** | **2654448106** |
| cuda-11.8 | 2654448107 | 2654448107 | 2654448107 | 2654448107 |

## Reduce 结果

- 原始 584 行 / 40071 字节，当前 25 行 / 915 字节。
- launch 缩为 `n=2, grid=(1), block=(2)`，只输出 `tid == 1` 的结果。
- 差分插桩显示 `_fv` 在 `_fv ^= (unsigned)v_22` 这一句开始分歧，进一步定位到
  `v_22 = (unsigned char)__byte_perm(...)` 到 `bool` 的转换。
- 最终核心为：

  ```cuda
  bool v_22 = ((unsigned char)__byte_perm(0u, input[GPU_SMOD(threadIdx.x, n)], 6u));
  unsigned _fv = input[GPU_SMOD(threadIdx.x, n)];
  _fv ^= ((unsigned)(v_22));
  if (threadIdx.x == 1u) output[0] = _fv;
  ```

## 根因

- NVVM 生成的 PTX 在三个工具链间语义一致，差异来自 **ptxas -O1 的 SASS
  lowering**。
- 正确路径（cuda-10.1 / cuda-11.8）：

  ```sass
  PRMT R8, RZ, 0x6, R6
  LOP3.LUT P0, RZ, R8, 0xff, RZ, 0xc0, !PT   // 重新计算 byte_perm 结果是否非零
  SEL  R0, RZ, 0x1, !P0
  ```

- cuda-11.4 `ptxas -O1`（O2/O3 行为相同）：

  ```sass
  PRMT RZ, RZ, 0x6, R4     // PRMT 结果被错误写回 RZ，等于丢弃
  SEL  R0, RZ, 0x1, !P0    // 复用了前面 n==0 分支遗留的 P0
  ```

  即 ptxas 错误删除了 `__byte_perm(...) & 0xff != 0` 的谓词计算，并复用了
  旧谓词 P0，导致 `bool v_22` 在 tid=1 时由 true 变成 false。
- O0 不触发，是因为 O0 保留完整的数据流；O1 开始做该错误优化。

### 更详细的数据流

最终样例对应的 PTX（三个工具链一致，节选）：

```ptx
setp.eq.s32 %p1, %r4, 0;          // 来自 GPU_SMOD 的 n==0 guard
...
prmt.b32 %r10, %r9, %r2, %r8;     // __byte_perm(0, input, 6)
and.b32  %r11, %r10, 255;
setp.ne.s32 %p3, %r11, 0;         // 正确路径应使用这个谓词
selp.u32 %r12, 1, 0, %p3;         // bool v_22 = (byte_perm_result != 0)
```

cuda-11.4 `ptxas -O1` 生成的 SASS 中：

```sass
ISETP.NE.AND P0, PT, RZ, c[0x0][0x170], PT ;  // P0 = (n != 0)
...
PRMT RZ, RZ, 0x6, R4                          // byte_perm 结果写入 RZ，被丢弃
SEL  R0, RZ, 0x1, !P0                         // 错误复用 P0 = (n != 0)
```

本样例中 `n == 2`，因此旧 P0 恒为 true，`!P0` 恒为 false，导致
`v_22` 被固定为 0（false）。正确值应由 `byte_perm(0, input[1], 6) & 0xff`
是否为非零决定，本样例中该值为 true。

对照 cuda-10.1 / cuda-11.8 的 O1 SASS：

```sass
PRMT R8, RZ, 0x6, R6
LOP3.LUT P0, RZ, R8, 0xff, RZ, 0xc0, !PT  // 由 byte_perm 结果重算谓词
SEL  R0, RZ, 0x1, !P0
```

因此根因可进一步细化为：cuda-11.4 ptxas -O1 将 `PRMT` 的目标寄存器错误
分配为 `RZ`，并随后删除了 `and + setp.ne` 这组谓词重算指令，造成
`SEL` 依赖了一个生命周期已经结束的旧谓词。
