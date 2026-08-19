#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "test.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z11kernel_mainPj(unsigned *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z11kernel_mainPj(unsigned *__par0){__cudaLaunchPrologue(1);__cudaSetupArgSimple(__par0, 0UL);__cudaLaunch(((char *)((void ( *)(unsigned *))kernel_main)));}
# 4 "test.cu"
void kernel_main( unsigned *__cuda_0)
# 4 "test.cu"
{__device_stub__Z11kernel_mainPj( __cuda_0);
# 10 "test.cu"
}
# 1 "asm/cuda-13.3_O0/test.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T0) {  __nv_dummy_param_ref(__T0); __nv_save_fatbinhandle_for_managed_rt(__T0); __cudaRegisterEntry(__T0, ((void ( *)(unsigned *))kernel_main), _Z11kernel_mainPj, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
