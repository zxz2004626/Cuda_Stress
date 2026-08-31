#include<cstdio>
__managed__ int m=1;
__global__ void k(int*i){
    auto f=threadIdx.y,t=blockIdx.x,a=t|1,b=t|3,c=t+5,d=t+7,e=t|5;
    char x=*i-129;__ldg(i);
    *i?f^=3*x:f^=__umul24(t,x>-2?1:x);
    printf("checksum=%u",e/c^a/d^f%b);
}

main(){
    k<<<1,1>>>(&m);
    cudaDeviceReset();
}