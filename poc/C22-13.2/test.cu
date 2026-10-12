// ============================================================================
// C22 最小触发用例 (minimized 2026-09-21, from test.cu deep-dive)
//
// 根因: cicc/NVVM (CUDA 11.5..13.2) 把 (unsigned long long)__mul24(x, x) % C
//       中的 __mul24 int 结果按"精确乘积"建模(等效零扩展)并删除 % C,
//       丢掉了 (u64)(int) 必须的符号扩展。CUDA 13.3 前端已改为正确 lowering
//       (cvt.s64.s32 + 魔数取模)。
//
// x = 6428759 时: __mul24 结果 = (int)0xA4EC6D91 = -1528009327 (负数)
//   正确 (ISO): (u64)(-1528009327) % 4986325391324574397
//               = 0x30670A6CA7B0155A → 截断 u32 = 0xA7B0155A
//   nvcc 错误:  取模被删 → (u32)0xA4EC6D91 = 0xA4EC6D91
//
// 触发条件: x 必须是运行时值(字面常量会被正确常量折叠, 13.2 上也不触发);
//           x 满足 (x*x) 截断到 int 后为负(本例 0x3039... 任意使
//           (x*x) & 0x80000000 != 0 的 24 位值均可, 如 x >= 46341 一类)。
//
// 判定: gpu == a7b0155a → CORRECT (13.3 / -G)
//       gpu == a4ec6d91 → BUG    (11.5..13.2 的 -O0/-O1/-O2/-O3)
//
// 输出格式: 先打印 checksum=<hex> (Makefile 由此抓取每级结果),
//           再打印人类可读的判定行。
// 版本窗口 (sm_86 实测): 11.5/11.8/12.2/12.5/12.8/13.0/13.1/13.2 → BUG;
//                        13.3 (全部 -O 等级) 与任意版本 -G → CORRECT。
// ============================================================================
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel_main(unsigned int* out, int x) {
    out[0] = (unsigned int)(((unsigned long long)__mul24(x, x)) % 4986325391324574397ull);
}

int main() {
    unsigned int* d;
    cudaMalloc(&d, 4);
    kernel_main<<<1, 1>>>(d, 6428759);
    cudaDeviceSynchronize();
    unsigned int v;
    cudaMemcpy(&v, d, 4, cudaMemcpyDeviceToHost);
    printf("checksum=%08x\n", v);
    printf("gpu=%08x  correct=a7b0155a  buggy=a4ec6d91  -> %s\n", v,
           v == 0xa7b0155a ? "CORRECT" : (v == 0xa4ec6d91 ? "BUG" : "UNEXPECTED"));
    cudaFree(d);
    return 0;
}
