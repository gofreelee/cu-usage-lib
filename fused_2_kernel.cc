#include <hip/hip_runtime.h>
#include <iostream>
#define HIP_ENABLE_PRINTF

__device__ void vecadd(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta);
__global__ void proxy_kernel(float *A, float *B, float *C, float *A_, float *B_, float *C_, int mask);
__device__ void vecadd_be(float *A, float *B, float *C, int start_cu_offset, int be_block_offset);
int main()
{
    float *host_a = (float *)malloc(sizeof(float) * 32 * 32);
    float *host_b = (float *)malloc(sizeof(float) * 32 * 32);
    float *host_c = (float *)malloc(sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i)
    {
        host_a[i] = 1;
        host_b[i] = 2;
        host_c[i] = 0;
    }
    float *device_a;
    float *device_b;
    float *device_c;
    hipMalloc((float **)&device_a, sizeof(float) * 32 * 32);
    hipMalloc((float **)&device_b, sizeof(float) * 32 * 32);
    hipMalloc((float **)&device_c, sizeof(float) * 32 * 32);

    hipMemcpy(device_a, host_a, sizeof(float) * 32 * 32, hipMemcpyHostToDevice);
    hipMemcpy(device_b, host_b, sizeof(float) * 32 * 32, hipMemcpyHostToDevice);

    float *host_a_ = (float *)malloc(sizeof(float) * 32 * 32);
    float *host_b_ = (float *)malloc(sizeof(float) * 32 * 32);
    float *host_c_ = (float *)malloc(sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i)
    {
        host_a_[i] = 1;
        host_b_[i] = 2;
        host_c_[i] = 0;
    }
    float *device_a_;
    float *device_b_;
    float *device_c_;
    hipMalloc((float **)&device_a_, sizeof(float) * 32 * 32);
    hipMalloc((float **)&device_b_, sizeof(float) * 32 * 32);
    hipMalloc((float **)&device_c_, sizeof(float) * 32 * 32);

    hipMemcpy(device_a_, host_a_, sizeof(float) * 32 * 32, hipMemcpyHostToDevice);
    hipMemcpy(device_b_, host_b_, sizeof(float) * 32 * 32, hipMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(60);
    // vecadd<<<grid, block>>>(device_a, device_b, device_c, M, N, K, alpha, beta);
    //  hipDeviceSynchronize();
    proxy_kernel<<<grid, block>>>(device_a, device_b, device_c, device_a_, device_b_, device_c_, 0);
    hipDeviceSynchronize();

    proxy_kernel<<<grid, block>>>(device_a, device_b, device_c, device_a_, device_b_, device_c_, 28);
    hipMemcpy(host_c_, device_c_, sizeof(float) * 32 * 32, hipMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i)
    {
        if (host_c_[i] != 3)
            std::cout << host_c_[i] << std::endl;
    }
}

__device__ void vecadd(float *A, float *B, float *C)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

__global__ void proxy_kernel(float *A, float *B, float *C, float *A_, float *B_, float *C_, int be_block_offset)
{
    if (blockIdx.x < 32)
    {
        vecadd(A, B, C);
    }
    else
    {
        //printf("debug\n");
        vecadd_be(A_, B_, C_, 32, be_block_offset);
    }
}

__device__ void vecadd_be(float *A, float *B, float *C, int start_cu_offset, int be_block_offset)
{
    if ((blockIdx.x) >= start_cu_offset)
    {
        if ((blockIdx.x - start_cu_offset) + be_block_offset < 32)
        {
            int idx = (blockIdx.x - start_cu_offset+ be_block_offset) * blockDim.x + threadIdx.x;
            C[idx] = A[idx] + B[idx];
        }
    }
    // if(blockIdx.x >= 31){
    //     printf("vecadd_be\n");
    // }
}
