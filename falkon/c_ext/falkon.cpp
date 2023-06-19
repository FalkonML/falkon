#include "falkon.h"

#include <torch/library.h>

#ifdef WITH_CUDA
#include <cuda.h>
#endif

namespace falkon {
    int64_t cuda_version() {
        #ifdef WITH_CUDA
            return CUDA_VERSION;
        #else
            return -1;
        #endif
    }

    TORCH_LIBRARY_FRAGMENT(falkon, m) {
        m.def("_cuda_version", &cuda_version);
    }
} // namespace falkon
