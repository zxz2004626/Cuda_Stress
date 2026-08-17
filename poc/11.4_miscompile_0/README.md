# 11.4_miscompile_0

自包含 CUDA-C 复现样例，从以下用例继续 reduce 得到：

`poc/cuda-11.4_miscompile/generator_ir__6aea4c8d5caa03cf/test.cu`

不使用 inline PTX；移位均带 guard，无源码 UB。

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`，当前 20 行 / 711 字节）
- `build_all.sh`
- `run_all.sh`

## 观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 1073741824 | 1073741824 | 1073741824 | 1073741824 |
| cuda-11.4 | **3221225472** | **3221225472** | **3221225472** | **3221225472** |
| cuda-11.8 | 1073741824 | 1073741824 | 1073741824 | 1073741824 |

## Reduce 结果

原始 490 行 / 32430 字节，当前 20 行 / 711 字节。进一步 reduce 时发现：

- 真正开始分歧的变量是循环前的 `v_6`，而不是先前 README 记录的 `v_19`；
  `v_19` 只是把 `v_6` 的差异传播到了输出。
- 因此改为直接输出 `v_6`，随后删除了原始 kernel 中所有为制造
  register-pressure / loop / helper 而保留的“上下文”语句。
- 最终触发条件压缩为两行核心表达式：

  1. 一个带 guard 的 64 位 `GPU_SSHL(0, variable_shift)`，使 10.1 与 11.4
     对 `v_5` 选择不同的 32/64 位 lowering；
  2. `(unsigned long long)(signed int) << 31`，cuda-11.4 将其错误 lowering
     为 `mul.wide.s32 ..., -2147483648`。

- launch 缩为 `grid=(1), block=(1)`；host 只分配/读取一个 `unsigned int`。

## 根因

- 最终 kernel 的 PTX 差异：

  - cuda-10.1：`cvt.s64.s32` + `bfe/bfi` 组合，结果正确。
  - cuda-11.4：`mul.wide.s32 %rd, var, -2147483648`，把无符号 64 位左移
    错误地用有符号 32x32->64 乘法实现，符号扩展导致高 32 位错误。
  - cuda-11.8：`cvt.s64.s32` + `shl.b64`，结果正确。

- 对于 `var_1 = 1`，正确结果为 `(1ULL << 31) >> 1` 的低 32 位 `0x40000000`；
  cuda-11.4 给出 `0xc0000000`。
- 单独写 `(unsigned long long)var_1 << 31` 不会触发；需要前面
  `GPU_SSHL(0, (blockIdx.x + 10))` 这类的 guarded-shift 上下文，10.1 才会走
  正确的 64 位路径，而 11.4 会提前截断为 32 位并错误选择 `mul.wide.s32`。
