#include "cublas_bindings.h"
#include "cusolver_bindings.h"
#include "../helpers.h"
#include "cuda_helpers.cuh"

#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <set>
#include <stdio.h>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>



namespace falkon {
namespace ops {
namespace {

//#define DEBUG 1

/* Input data structures */
struct blockAlloc {
    int start;
    int end;
    int size;
    int device;
    int id;
};

/* CUDA CallBacks */
struct callBackData {
    std::atomic<int> *work_unit;
    const int x;
    const int y;
    const int callee;
};

void CUDART_CB copyCallBack(cudaStream_t stream, cudaError_t error, void *data) {
    callBackData *tmp = (callBackData *)(data);
#ifdef DEBUG
    fprintf(stderr, "Incrementing work unit at [%d, %d] callee: %d - from %d\n", tmp->x, tmp->y, tmp->callee, tmp->work_unit->load());
#endif
    std::atomic_fetch_add(tmp->work_unit, 1);
}

/* Data-loading helper functions */
template <typename scalar_t>
static inline void load_block(
        at::Tensor &data_h,
        scalar_t*     &data_d,
        const blockAlloc& alloc_i,
        const blockAlloc& alloc_j,
        const int mbs,
        const cudaStream_t stream)
{
    const int64_t si = data_h.stride(0);
    const int64_t sj = data_h.stride(1);
    scalar_t *data_h_ptr = data_h.data_ptr<scalar_t>();
    const uint64_t offset = si * alloc_i.start + sj * alloc_j.start;
    FLK_CUDABLAS_CHECK(cublasSetMatrixAsync(
        /*rows=*/alloc_i.size,
        /*cols=*/alloc_j.size,
        /*elem_size=*/sizeof(scalar_t),
        /*A=*/(void *)(data_h_ptr + offset),
        /*lda=*/sj,
        /*B=*/(void *)data_d,
        /*ldb=*/mbs,
        /*stream=*/stream
    ));
}
template <typename scalar_t>
static inline void get_block(
        scalar_t*     &data_d,
        at::Tensor &data_h,
        const blockAlloc& alloc_i,
        const blockAlloc& alloc_j,
        const int mbs,
        const cudaStream_t stream)
{
    const int64_t si = data_h.stride(0);
    const int64_t sj = data_h.stride(1);
    scalar_t *data_h_ptr = data_h.data_ptr<scalar_t>();
    const uint64_t offset = si * alloc_i.start + sj * alloc_j.start;
    FLK_CUDABLAS_CHECK(cublasGetMatrixAsync(
        /*rows=*/alloc_i.size,
        /*cols=*/alloc_j.size,
        /*elem_size=*/sizeof(scalar_t),
        /*A=*/(void *)data_d,
        /*lda=*/mbs,
        /*B=*/(void *)(data_h_ptr + offset),
        /*ldb=*/sj,
        /*stream=*/stream
    ));
}
template <typename scalar_t>
static inline void opt_load_block(
        at::Tensor &data_h,
        scalar_t*     &data_d,
        const int block_id,
        std::set<int> &col0_fill,
        const blockAlloc& alloc_i,
        const blockAlloc& alloc_j,
        const int mbs,
        const cudaStream_t stream)
{
    if (col0_fill.find(block_id) == col0_fill.end()) {
        load_block<scalar_t>(data_h, data_d, alloc_i, alloc_j, mbs, stream);
        col0_fill.insert(block_id);
    }
}

/* Main parallel POTRF function */
void parallel_potrf_runner(
        int device_id,
        std::vector<std::vector<std::atomic<int>>> &work,
        at::Tensor &A,
        std::vector<blockAlloc> &allocs) {
    // CUDA devices and stream
    const at::cuda::CUDAGuard g(device_id);
    at::cuda::CUDAStream s1 = at::cuda::getStreamFromPool(false, device_id);
    at::cuda::CUDAStream s2 = at::cuda::getStreamFromPool(false, device_id);
    at::cuda::CUDAStream s3 = at::cuda::getStreamFromPool(false, device_id);
    c10::cuda::CUDAStreamGuard g0(s1);
    // Fetch cuBLAS handle and set cuBLAS, cuSOLVER streams to s1 (automatically done by the getCurrentHandle code)
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    auto cusolver_handle = at::cuda::getCurrentCUDASolverDnHandle();

    cudaStream_t s1_c = s1.stream();
    cudaStream_t s2_c = s2.stream();
    cudaStream_t s3_c = s3.stream();

    const auto scalar_type = A.scalar_type();
    const int k = allocs.size();

    const int mbs = (*std::max_element(allocs.begin(), allocs.end(), [] (blockAlloc lhs, blockAlloc rhs) {
        return lhs.size < rhs.size;
    })).size;
    const uint64_t mbs_sq = mbs*mbs;
    // Figure out `my_blocks` the blocks of the current stage
    std::vector<blockAlloc> my_blocks;
    std::set<int> my_block_ids;
    for (auto &block : allocs) {
        if (block.device == device_id) {
            my_blocks.push_back(block);
            my_block_ids.insert(block.id);
        }
    }
    std::map<std::pair<int, int>, callBackData> callback_data;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            const callBackData cback_data = {.work_unit = &(work[i][j]), .x = i, .y = j, .callee = -1};
            callback_data.insert(std::pair<std::pair<int, int>, callBackData>(std::pair<int, int>(i, j), cback_data));
        }
    }

    // col0_fill keeps track of the 'current' column: which blocks are loaded or not.
    std::set<int> col0_fill;

    // First GPU buffer allocation.
    const uint64_t buf_size = mbs_sq * (k + k + 1);
    const auto buf_opt = at::TensorOptions()
        .dtype(A.dtype())
        .device(at::kCUDA, device_id)
        .layout(at::kStrided)
        .requires_grad(false);
    const auto data_buf = at::empty(buf_size, buf_opt);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch_parallel_potrf", [&] {
    const scalar_t mone = -1.0;
    const scalar_t one = 1.0;

    // How much workspace does potrf need:
    int potrf_buf_size = 0;
    potrf_buffersize<scalar_t>(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        /*n=*/mbs,
        A.data_ptr<scalar_t>(),
        /*lda=*/mbs,
        &potrf_buf_size);
    const auto potrf_buf = at::empty(potrf_buf_size, buf_opt);
    const auto potrf_info_buf = at::zeros(1, at::dtype(at::kInt).device(at::kCUDA, device_id));

    // Data buffers
    scalar_t *data_buf_ptr = data_buf.data_ptr<scalar_t>();
    scalar_t *potrf_buf_ptr = potrf_buf.data_ptr<scalar_t>();
    int *potrf_info_buf_ptr = potrf_info_buf.data_ptr<int>();
    scalar_t *col0_h[k];
    for (int i = 0; i < k; i++) {
        col0_h[i] = data_buf_ptr;
        data_buf_ptr += mbs_sq;
    }
    scalar_t *col1_h[k];
    for (int i = 0; i < k; i++) {
        col1_h[i] = data_buf_ptr;
        data_buf_ptr += mbs_sq;
    }
    scalar_t *g_buf = data_buf_ptr;

    // Book-keeping variables (used in the loop)
    uint col_updates_left;
    uint trail_updates_left;
    int potrf_info_h = 0;
    scalar_t **col_buf_h;
    scalar_t **next_buf_h;
    cudaStream_t s_copyback;
    // Start the main loop
    for (int i = 0; i < k; i++) {
#ifdef DEBUG
        fprintf(stderr, "Starting iteration %d\n", i);
#endif
        // Setup double-buffering (via pre-inserting elements in col0_fill)
        // and number of updates.
        col_updates_left = 0;
        trail_updates_left = 0;
        for (const auto& mb : my_blocks) {
            if (mb.id > i) {
                col_updates_left += 1;
                trail_updates_left += mb.id - i;
                if (i != 0) {col0_fill.insert(mb.id);}
            }
        }
        // Switch the double-buffered col0, col1
        if (i % 2 == 0) {
            col_buf_h = col0_h;
            next_buf_h = col1_h;
            s_copyback = s2_c;
        } else {
            col_buf_h = col1_h;
            next_buf_h = col0_h;
            s_copyback = s3_c;
        }
        C10_CUDA_CHECK(cudaStreamSynchronize(s_copyback));

        // 1. POTRF
        scalar_t * i_block = col_buf_h[i];
        const auto& i_alloc = allocs[i];
        if (i_alloc.device == device_id) {
            while (work[i][i] != i) { std::this_thread::yield(); }
            opt_load_block<scalar_t>(A, i_block, i, col0_fill, i_alloc, i_alloc, mbs, s1_c); // [i, i]
            potrf<scalar_t>(cusolver_handle, CUBLAS_FILL_MODE_LOWER, /*n=*/i_alloc.size,
                            /*A=*/i_block, /*lda=*/mbs, /*work=*/potrf_buf_ptr, /*lwork=*/potrf_buf_size,
                            potrf_info_buf_ptr);
            C10_CUDA_CHECK(cudaStreamSynchronize(s1_c));
            if (potrf_info_h != 0) {
                AT_ERROR("Cholesky decomposition failed: leading minor of order ",
                         potrf_info_h, " is not positive definite.");
            }
            get_block<scalar_t>(i_block, A, i_alloc, i_alloc, mbs, s_copyback);
            C10_CUDA_CHECK(cudaStreamAddCallback(s_copyback, copyCallBack, &callback_data.at(std::pair<int, int>(i, i)), 0));
#ifdef DEBUG
            fprintf(stderr, "D:%d  Iteration %d stage %d - finished [%d, %d]\n", device_id, i, 1, i, i);
#endif
        }

        // 2. COLUMN UPDATE
        while (work[i][i] < i + 1) { std::this_thread::yield(); }
        // Keep track of which blocks we have already processed.
        // work table cannot work for this here, since it is set asynchronously.
        std::unordered_set<int> processed_idx;
        while (col_updates_left > 0) {
            for (const auto& b_alloc : my_blocks) {
                const int b = b_alloc.id;
                if (b <= i || processed_idx.find(b) != processed_idx.end() || work[b][i] != i) {
                    continue;
                }
                scalar_t *b_block = col_buf_h[b];

                opt_load_block<scalar_t>(A, i_block, i, col0_fill, i_alloc, i_alloc, mbs, s1_c); // [i, i]
                opt_load_block<scalar_t>(A, b_block, b, col0_fill, b_alloc, i_alloc, mbs, s1_c); // [b, i]
                falkon::ops::trsm<scalar_t>(
                    cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    b_alloc.size, i_alloc.size, &one, i_block, mbs, b_block, mbs);
                C10_CUDA_CHECK(cudaStreamSynchronize(s1_c));

                get_block<scalar_t>(b_block, A, b_alloc, i_alloc, mbs, s_copyback);
                C10_CUDA_CHECK(cudaStreamAddCallback(s_copyback, copyCallBack, &callback_data.at(std::pair<int, int>(b, i)), 0));

                col_updates_left--;
                processed_idx.insert(b);
#ifdef DEBUG
                fprintf(stderr, "D:%d  Iteration %d stage %d - finished [%d, %d]\n", device_id, i, 2, b, i);
#endif
            }
        }

        // 3. TRAILING UPDATE
        // Note that this loop does not need `processed_idx` like loop 2
        // since it is processed in order. In fact the outer while loop
        // is unnecessary
#ifdef DEBUG
        fprintf(stderr, "Starting stage 3\n");
#endif
        while (trail_updates_left > 0) {
            for (const auto& b_alloc : my_blocks) {
                int b = b_alloc.id;
                if (b < i + 1) { continue; }
                while (work[b][i] != i + 1) { std::this_thread::yield(); }

                scalar_t * b_block = col_buf_h[b];
                for (int y = b; y > i; y--) {
                    while (work[y][i] != i + 1 || work[b][y] != i) { std::this_thread::yield(); }
                    const auto& y_alloc = allocs[y];
                    scalar_t *y_block = col_buf_h[y];
                    opt_load_block<scalar_t>(A, y_block, y, col0_fill, y_alloc, i_alloc, mbs, s1_c); // [y, i]
                    load_block<scalar_t>(A, g_buf, b_alloc, y_alloc, mbs, s1_c); // [b, y]
                    if (b_alloc.id != y_alloc.id) {
                        gemm<scalar_t>(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, b_alloc.size, y_alloc.size, i_alloc.size,
                                       &mone, b_block, mbs, y_block, mbs, &one, g_buf, mbs);
                    } else {
                        syrk<scalar_t>(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, b_alloc.size,
                                       i_alloc.size, &mone, b_block, mbs, &one, g_buf, mbs);
                    }
                    if (y == i + 1) {
                        // We are on the column which will be tackled next, can copy directly to col0
                        C10_CUDA_CHECK(
                            cudaMemcpyAsync(next_buf_h[b], g_buf, mbs_sq * sizeof(scalar_t),
                            cudaMemcpyDeviceToDevice, s1_c));
                        C10_CUDA_CHECK(cudaStreamSynchronize(s1_c));
                        get_block<scalar_t>(next_buf_h[b], A, b_alloc, y_alloc, mbs, s_copyback);
                        C10_CUDA_CHECK(cudaStreamAddCallback(s_copyback, copyCallBack, &callback_data.at(std::pair<int, int>(b, y)), 0));
                    } else {
                        // We must free the `g_buf` variable before the next round.
                        get_block<scalar_t>(g_buf, A, b_alloc, y_alloc, mbs, s1_c);
                        C10_CUDA_CHECK(cudaStreamSynchronize(s1_c));
                        std::atomic_fetch_add(&work[b][y], 1);
                    }
                    trail_updates_left--;
#ifdef DEBUG
                    fprintf(stderr, "D:%d  Iteration %d stage %d - finished [%d, %d]\n", device_id, i, 3, b, y);
#endif
                }
            }
        }
        col0_fill.clear();
    }
    C10_CUDA_CHECK(cudaStreamSynchronize(s1_c));
    C10_CUDA_CHECK(cudaStreamSynchronize(s2_c));
    C10_CUDA_CHECK(cudaStreamSynchronize(s3_c));
    });  // end dispatch float
}

at::Tensor parallel_potrf_kernel(
        c10::IntArrayRef devices,
        std::vector<blockAlloc> allocations,
        at::Tensor &A) {
    CHECK_CPU(A);
    // Initialize the atomic table
    int k = allocations.size();
    std::vector<std::vector<std::atomic<int>>> work(k);
    for (int i = 0; i < k; i++) {
        work[i] = std::vector<std::atomic<int>>(k);
        for (int j = 0; j < k; j++) {
            work[i][j].store(0);
        }
    }

    std::vector<std::thread> threads;
    for (const int64_t device_id : devices) {
        threads.push_back(
            std::thread(&parallel_potrf_runner, device_id, std::ref(work), std::ref(A), std::ref(allocations)));
    }

    for (auto& t : threads) {
        t.join();
    }
    return A;
}

at::Tensor parallel_potrf(
     c10::IntArrayRef devices,
     c10::IntArrayRef block_starts,
     c10::IntArrayRef block_ends,
     c10::IntArrayRef block_sizes,
     c10::IntArrayRef block_devices,
     c10::IntArrayRef block_ids,
     at::Tensor& A) {
    std::vector<blockAlloc> out_allocs;
    for (const int i : c10::irange(block_starts.size())) {
        blockAlloc ba = {
            .start = (int)block_starts[i],
            .end   = (int)block_ends[i],
            .size  = (int)block_sizes[i],
            .device= (int)block_devices[i],
            .id    = (int)block_ids[i]
        };
        out_allocs.push_back(ba);
    }
    return parallel_potrf_kernel(devices, out_allocs, A);
}

} // namespace


TORCH_LIBRARY_IMPL(falkon, CPU, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("falkon::parallel_potrf"),
        TORCH_FN(parallel_potrf));
}

} // namespace ops
} // namespace falkon
