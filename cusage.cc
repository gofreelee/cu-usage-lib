#include "cusage.h"

int get_kernel_resource(hipFunction_t hip_func, kernel_resource &ret)
{
    hipFunctionWGInfo_t wg_info;
    GPU_RETURN_STATUS(hipFuncGetWGInfo(hip_func, &wg_info));
    ret.shared_memory = wg_info.usedLDSSize_;
    ret.vgprs = wg_info.usedVGPRs_;
    ret.sgprs = wg_info.usedSGPRs_;
    ret.stack_size = wg_info.privateMemSize_;
    return Status::Succ;
}

int calculate_occupancy(const kernel_resource &resource, dim3 block_dim)
{
    int vgprs = align_up(resource.vgprs, 4);
    int sgprs = align_up(resource.sgprs, 8);
    int shared_mem = align_up(resource.shared_memory, 256);
    int block_size = (int)align_up<unsigned int>(block_dim.x * block_dim.y * block_dim.z, 64);

    int max_gpr_waves = (16 * 1024 / (vgprs * 64)) * 4;
    max_gpr_waves = std::min(max_gpr_waves, (800 / sgprs) * 4);
    max_gpr_waves = std::min(max_gpr_waves, 40);

    int max_gpr_blocks = max_gpr_waves * 64 / block_size;
    int max_shared_mem_blocks = 64 * 1024 / block_size;

    int max_thread_blocks = 2048 / block_size;

    int occupancy = std::min(max_gpr_blocks, max_shared_mem_blocks);
    occupancy = std::min(occupancy, max_thread_blocks);

    // std::cout << "occupancy is " << occupancy << std::endl;

    return occupancy;
}