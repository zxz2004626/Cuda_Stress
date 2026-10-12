# C46 — ptxas -O2/-O3 谓词 OR（bool 累积）miscompile

## TL;DR

`test.cu` 中一个**幂等的布尔累积** `v_24 |= cond;`（`cond` 循环不变、运行期非常量）被
**ptxas -O2/-O3 编错**：外层循环展开成 2 份拷贝后，第 2 份拷贝里的谓词 OR 累积被错编成
**“恒为 true”**（`PLOP3.LUT ..., 0xff`）而不是 `P1 = P1 | P0`（应为 LUT `0xae`/带 PT 的 `0xa8`）。
效果是：外层第 2 次迭代起 `v_24` 被无条件置 1。

- `-O0` / `-O1`：正确（与 CPU 参考实现一致）
- `-O2` / `-O3` / nvcc 默认：**错误**，受影响的 block 输出差 1
- 11.6 引入，**13.3 仍未修复**（11.1/11.3 正确）
- compute-sanitizer memcheck/initcheck/racecheck 全干净 → 纯优化器 bug，与内存/同步无关

最小触发样例：**[minimal.cu](minimal.cu)**（单 block 单线程，`n=1`，自带 CPU 参考实现，直接打印 match/MISMATCH）。

## 复现

```bash
cd C46
NVCC=/export/ssd1/xzhanglb/nvcc-toolkit/cuda-13.3/bin/nvcc
FLAGS="-std=c++17 -arch=sm_86 -w -ccbin /usr/bin/g++-10"

$NVCC $FLAGS -Xptxas=-O1 minimal.cu -o O1 && ./O1
#   gpu=234183369 reference=234183369 match
#   checksum=2888621862
$NVCC $FLAGS -Xptxas=-O2 minimal.cu -o O2 && ./O2
#   gpu=234183368 reference=234183369 MISMATCH
#   checksum=2888621857
```

原始样例 `test.cu`（`make`）同等复现：
`ptxas-O0 checksum=7338220735216791338`，`ptxas-O1 checksum=7338220735216791338`（正确），
`ptxas-O2 checksum=12305328915047689402`（错误）。

## 版本矩阵（minimal.cu，全部 sm_86，RTX 3090 实测）

| CUDA | 11.1 | 11.3 | 11.6 | 11.8 | 12.2 | 12.5 | 12.8 | 13.0 | 13.1 | 13.2 | 13.3 |
|------|------|------|------|------|------|------|------|------|------|------|------|
| O1   | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    |
| O2   | ✓    | ✓    | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** | **✗** |

与 metadata.yaml 归档的 trigger matrix 一致：**11.6 引入，unfixed as of 13.3**。
默认优化级别（ptxas -O3）同样出错。

## 代码模式与触发必要条件

源码中的关键形状（minimal.cu，与 test.cu 逐字对应）：

```cuda
bool v_24 = 0;
unsigned int v_29 = f(blockIdx.x);              // block 不变 → 谓词可外提但非常量
for (int _rep = 0; _rep < 2; _rep++) {          // 外层 2 次 → ptxas 展开成 2 份拷贝
    for (int i_0 = 0; i_0 < 16; i_0++) {        // 循环嵌套必须保留（cicc 对 16-trip 做 ×4 部分展开）
        int dw_1 = 0;
        do {
            v_24 |= (bool)(...)(short)v_29;     // ★ 幂等谓词 OR 累积（循环携带 bool）
            helper_3();                          // 挡 cicc LICM 的“不透明”操作（见下）
            dw_1++;
        } while (dw_1 < 4);
    }
    ... egres_0 = funnel-shift 混合 input ...    // 外层循环体的负载（见必要条件 4）
    _fv ^= (unsigned)v_24;
}
```

reduce 过程中逐一验证的触发必要条件：

| # | 要素 | 结论 |
|---|------|------|
| 1 | `v_24 |= cond`，`cond` 为循环不变、运行期非常量的谓词 | 必须。cicc 从 `(short)v_29 != 0` 经代数折叠生成 **64 位** `setp.ne.s64` 谓词（`%p3`）。换成等价的 32 位 `setp.ne.s32` 后不再触发 |
| 2 | 累积链的形状：`cvt.u32.u16` → **`cvt.s32.s8`**（符号扩展）→ `or` → `setp` → `or.pred` → `selp.u16` 回写 | 必须。去掉 s8 符号扩展这一步（语义等价）后 -O2 就编对了 |
| 3 | 双层循环：内层 4 次幂等 OR，外层 2 次 | 必须。bug 发生在外层第 2 份展开拷贝里 |
| 4 | 外层循环体含 3 个带除零守卫的 `input[]` load + `GPU_SSHR` 的移位守卫分支（PTX `@%p26 bra`） | 必须。删掉任一 load 或去掉该分支，ptxas 的展开/变换路径改变，不再触发 |
| 5 | 源码层 16×4 循环嵌套 | 必须。合并成单层 `k<4` 循环后 cicc 全展开，循环到不了 ptxas |
| 6 | `helper_3()`（funnelshift→inline asm + switch） | 仅源码层需要：阻止 cicc 把幂等累积 LICM 出循环。PTX 层直接编辑可完全删掉（见 `reduce/t4.ptx`，仍触发）|
| 7 | `var_1`/`var_2` 参数、helper_3 的 case 2、`if (~0ll<0ll)` 包装、GPU_SMUL 宏、输入 buffer ×32、grid/block/n 规模 | 不需要，已删（grid=1×block=1、n=1 即可，block 0 的 `cond` 恰为 false，输出差直接可见）|

## 定位过程与 SASS 证据

cicc 生成的 PTX 三个优化级别完全相同（`-Xptxas` 只影响 ptxas），即前端输出正确；
差异全部产生在 ptxas。以 13.3 为例（完整 SASS 见 `reduce/min_O2.sass`、`reduce/test_O2.sass`）：

**-O1（正确）**：内层循环保留为循环，每次迭代做一次谓词 OR：

```
/*0300*/  PLOP3.LUT P1, PT, P1, P0, PT, 0xa8, 0x0 ;   // P1 = P1 | P0        ✓
/*0310*/  SEL       R6, RZ, 0x1, !P1 ;                 // v_24 = P1 ? 1 : 0   ✓
```

**-O2（错误）**：外层 `_rep` 循环被完全展开成 2 份拷贝，每份含 4 个（内层展开的）累积点。

第 1 份拷贝 —— 正确的 OR：

```
/*07b0*/  ISETP.NE.U32.OR P1, PT, R4, RZ, P0 ;        // P1 = (v_24!=0) | P0   ✓
/*07c0*/  PLOP3.LUT P1, PT, P1, P0, PT, 0xa8, 0x0 ;   // P1 = P1 | P0          ✓
/*07d0*/  ISETP.NE.U32.OR P1, PT, R4, RZ, P1 ;        //   （×4 个累积点全部正确）
/*07e0*/  PLOP3.LUT P1, PT, P1, P0, PT, 0xa8, 0x0 ;
/*0840*/  SEL R6, RZ, 0x1, !P1 ;                      // v_24 = P1 ? 1 : 0     ✓
```

第 2 份拷贝 —— 同样的 OR 被编成“恒为 true”：

```
/*0eb0*/  PLOP3.LUT P1, PT, P0, !PT, P1, 0xff, 0x0 ;  // P1 = TRUE             ✗
/*0ee0*/  PLOP3.LUT P1, PT, P0, !PT, P1, 0xff, 0x0 ;  //   （×4 个累积点全部错误）
/*0f10*/  PLOP3.LUT P1, PT, P0, !PT, P1, 0xff, 0x0 ;
/*0f50*/  PLOP3.LUT P1, PT, P0, !PT, P1, 0xff, 0x0 ;
/*11c0*/  SEL R6, RZ, 0x1, !P1 ;                      // v_24 = 1（无条件）     ✗
```

`PLOP3.LUT` 的立即数是对 3 个布尔输入 (a,b,c) 的 8 位真值表：
- `0xa8` = `(a&b)|(a&c)|(b&c)`，当 c=PT 时即 `a|b` —— 语义正确的 `or.pred`；
- `0xff` = 全 1 —— **无论输入是什么结果恒为 true**。

即 ptxas -O2 在展开后的第 2 份拷贝中，把 `P1 = P1 | P0` 的真值表错算成了常量 1
（症状等价于把某个操作数错误地按“恒真”折叠进了 truth table）。于是该拷贝中
`v_24 := true`，最终 `_fv` 在第 2 次迭代多 XOR 了一个 1。

由于 `cond` 只依赖 `blockIdx.x`，错误只影响 `cond == false` 的 block（第 1 份拷贝的
v_24 本来就是 0、第 2 份被错误置 1）；`cond == true` 的 block 两种结果相同，因此 checksum
只有部分输出位翻转——这也是它没被常规对拍立即发现的原因。minimal.cu 选中 block 0
（`v_29 == 0` → `cond == false`），使差异 100% 显现。

## Root cause 结论

**ptxas -O2/-O3 的谓词优化（unroll 后的 PLOP3 谓词 OR 链常量折叠）存在编码错误**：
循环完全展开后，对“幂等 OR 累积”在非首份展开拷贝中生成 LUT=0xff（恒真）而非 OR 的
真值表（0xae / 带 PT 的 0xa8），把 `bool |= cond` 错误地变成 `bool = true`。
cicc 前端与 PTX 无辜（-O0/-O1 结果正确且与 CPU 参考一致）。该 bug 自 CUDA 11.6 引入，
历经 11.8/12.x/13.x 全部版本至 13.3 未修复。与已归档的 rem/div-prenot（12.2 已修）和
umul64hi（-O1+ 即触发）均不相同。

## 规避

- 编译时加 `-Xptxas=-O1`（或 `-O0`）；注意 nvcc **默认即 ptxas -O3**，必须显式降级。
- 或改写源码打破触发形状：把循环携带的 `bool` 改为 `unsigned` 直接 OR（改变累积链的
  cvt.s32.s8 形状），或去掉循环嵌套（但会失去 cicc 保循环的效果，需另行验证）。

## 文件

- `test.cu` / `Makefile` — 原始样例与三级别对拍
- `minimal.cu` — **最小触发样例**（自带 CPU 参考实现，直接输出 match/MISMATCH）
- `metadata.yaml` — 归档元数据
- `reduce/` — reduce 过程产物：
  - `check.sh` / `check2.sh` — 源码级差分（O0/O1/O2 checksum 对拍）
  - `checkptx.sh` + `driver.cu` — PTX 级差分（ptxas 直接编 cubin，driver API 加载运行，秒级迭代）
  - `t1/t3/t4/t5b/t6_counter.ptx` — 关键 PTX 约简阶段（t4 证明 inline-asm 在 PTX 层不必要）
  - `t6_{p3fold,accchain}.ptx` — 证明 p3 折叠与 s8 扩展为必要条件的反例
  - `test_O1.sass` / `test_O2.sass` / `min_O2.sass` — SASS 证据
  - `f2..f7.cu` — 源码级中间阶段
