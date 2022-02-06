#pragma once

#include <vector>
#include <torch/extension.h>
#include <cusolverDn.h>

struct blockAlloc {
    int start;
    int end;
    int size;
    int device;
    int id;
};

struct gpuInfo {
    float free_memory;
    int id;
};

torch::Tensor parallel_potrf_cuda(std::vector<gpuInfo> gpu_info,
                                  std::vector<blockAlloc> allocations,
                                  torch::Tensor &A);
