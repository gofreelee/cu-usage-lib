#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
enum Status {
    Succ,
    Fail,
    NotFound,
    OutOfRange,
    Full
};
template <typename T>
T align_up(T value, T alignment) {
    T temp = value % alignment;
    return temp == 0? value : value - temp + alignment;
}

#define GPU_RETURN_STATUS(cmd)                                                                               \
    {                                                                                                        \
        hipError_t error = cmd;                                                                              \
        if (error != hipSuccess)                                                                             \
        {                                                                                                    \
            std::cout << "hip error: " << hipGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
            return Status::Fail;                                                                             \
        }                                                                                                    \
    }

#define ASSERT_GPU_ERROR(cmd)                                                                                \
    {                                                                                                        \
        hipError_t error = cmd;                                                                              \
        if (error != hipSuccess)                                                                             \
        {                                                                                                    \
            std::cout << "hip error: " << hipGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    }

struct kernel_resource
{
    int shared_memory;
    int vgprs;
    int sgprs;
    int stack_size;
};

int get_kernel_resource(hipFunction_t hip_func, kernel_resource& ret_resource);

int calculate_occupancy(const kernel_resource& resource, dim3 block_dim);