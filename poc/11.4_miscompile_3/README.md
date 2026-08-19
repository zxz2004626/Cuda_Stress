# 11.4_miscompile_3

自包含 CUDA-C 复现样例，从以下用例持续 reduce 得到：

`poc/cuda-11.4_miscompile/generator_ir__18a58239ac1844f6/test.cu`

不使用 inline PTX；输入取模、移位宽度和全局数组索引均不会越界/溢出。

## 文件

- `runtime_minimal.cu`（硬链接别名 `test.cu`）
- `build_all.sh`
- `run_all.sh`

## 观察结果

| compiler | O0 | O1 | O2 | O3 |
| --- | --- | --- | --- | --- |
| cuda-10.1 | 2419750549 | 2419750549 | 2419750549 | 2419750549 |
| cuda-11.4 | 2419750549 | 2419750549 | **2419750548** | **2419750548** |
| cuda-11.8 | 2419750549 | 2419750549 | 2419750549 | 2419750549 |

## Reduce 结果

- 原始 760 行 / 49655 字节，当前 **37 行 / 1421 字节**。
- 输出从 32768 个元素 checksum 缩为 `n=1, grid=(1), block=(1)`，只保留 `tid == 0` 的 checksum。
- 删除了 helper、shared memory、`__launch_bounds__`、switch、一个 atomicMin 以及大量只影响寄存器压力的中间变量。
- 最终 kernel 只保留：
  - 一次 input 加载；
  - 一个零初始化全局数组 `arr_0` 上的 `__byte_perm` / `__brevll` /
    `__funnelshift_*` 数据流；
  - 一个 3 次循环，循环内为一次 `atomicMin` 和 7 项 `_fv` XOR 累加。

## Root cause

- NVVM 生成的 PTX 在 O0/O1/O2/O3 下完全一致（仅 `ptxasOptions` 不同），
  因此错误位于 **cuda-11.4 `ptxas -O2`**。
- O2 将 3 次循环展开后，把 `atomicMin` 与其谓词/地址计算穿插在 XOR
  累加链之间。比较 SASS 尾部可见最终 LOP3 的输入寄存器在 O1/O2 下不同：

  cuda-11.4 O1（正确）：

  ```sass
  LOP3.LUT R2, R3, R2, R7, 0x96
  LOP3.LUT R3, R13, R9, R2, 0x96
  LOP3.LUT R4, R5, R4, R2, 0x96
  LOP3.LUT R3, R3, R4, RZ, 0x3c
  STG.E.SYS [UR4], R3
  ```

  cuda-11.4 O2（错误）：

  ```sass
  LOP3.LUT R6, R7, R6, RZ, 0x3c
  STG.E.SYS [UR4], R6
  ```

  O2 在展开后的 atomicMin/XOR 数据流中复用了错误的活动寄存器，导致最终
  XOR 少翻转 bit0，checksum 从 `...549` 变为 `...548`。
- cuda-11.4 O1 未做该展开/重排，结果正确；O3 继承 O2 的错误调度。
