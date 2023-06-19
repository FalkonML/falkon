#include "../helpers.h"

#include <optional>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/macros/Macros.h>

namespace falkon {
namespace ops {
namespace {

#define ASSERT_IS_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")


template <typename scalar_t>
static inline void sparse_bdot_impl(
        scalar_t* data1,
        int64_t* indices1,
        int64_t* indptr1,
        scalar_t* data2,
        int64_t* indices2,
        int64_t* indptr2,
        scalar_t* out_data,
        int64_t N) {
    // row start and row end for both input matrices
    int64_t rs1, re1, rs2, re2;
    // column indices (in the `indices1` and `indices2` arrays)
    int64_t colidx1, colidx2;
    int64_t i;

    for (i = 0; i < N; i++) {
        rs1 = indptr1[i];
        re1 = indptr1[i + 1];
        rs2 = indptr2[i];
        re2 = indptr2[i + 1];

        while (rs1 < re1 && rs2 < re2) {
            colidx1 = indices1[rs1];
            colidx2 = indices2[rs2];
            if (colidx1 < colidx2) {
                rs1++;
            } else if (colidx1 > colidx2) {
                rs2++;
            } else {
                out_data[i] += data1[rs1] * data2[rs2];
                rs1++;
                rs2++;
            }
        }
    }
}


at::Tensor sparse_bdot_kernel(
        const at::Tensor &indexptr1,
        const at::Tensor &indices1,
        const at::Tensor &data1,
        const at::Tensor &indexptr2,
        const at::Tensor &indices2,
        const at::Tensor &data2,
        at::Tensor &out) {
    ASSERT_IS_CPU(indexptr1);
    ASSERT_IS_CPU(indices1);
    ASSERT_IS_CPU(data1);
    ASSERT_IS_CPU(indexptr2);
    ASSERT_IS_CPU(indices2);
    ASSERT_IS_CPU(data2);
    ASSERT_IS_CPU(out);
    AT_ASSERTM(indexptr1.dim() == 1, "indexptr must be 1D");
    AT_ASSERTM(indexptr1.stride(0) == 1, "indexptr must be memory-contiguous");
    AT_ASSERTM(indices1.dim() == 1, "indices must be 1D");
    AT_ASSERTM(indices1.stride(0) == 1, "indices must be memory-contiguous");
    AT_ASSERTM(data1.dim() == 1, "data must be 1D");
    AT_ASSERTM(data1.stride(0) == 1, "data must be memory-contiguous");
    AT_ASSERTM(indexptr2.dim() == 1, "indexptr must be 1D");
    AT_ASSERTM(indexptr2.stride(0) == 1, "indexptr must be memory-contiguous");
    AT_ASSERTM(indices2.dim() == 1, "indices must be 1D");
    AT_ASSERTM(indices2.stride(0) == 1, "indices must be memory-contiguous");
    AT_ASSERTM(data2.dim() == 1, "data must be 1D");
    AT_ASSERTM(data2.stride(0) == 1, "data must be memory-contiguous");
    AT_ASSERTM(indexptr1.size(0) == indexptr2.size(0), "the two sparse matrices must have the same number of rows.");
    AT_ASSERTM(data1.scalar_type() == data2.scalar_type(), "the two sparse matrices must be of the same type.");
    AT_ASSERTM(out.scalar_type() == data1.scalar_type(), "Matrices A, B and out must be of the same type.");

    int64_t N = indexptr1.size(0) - 1;
    AT_ASSERTM(N == out.size(0), "Input shape mismatch");
    if (out.dim() >= 2) {
        AT_ASSERTM(out.size(1) == 1, "Output array must be 1D");
    }
    out.fill_(0.0);

    auto scalar_type = data1.scalar_type();

    AT_DISPATCH_ALL_TYPES(scalar_type, "sparse_bdot_impl", [&] {
        sparse_bdot_impl<scalar_t>(
                data1.data_ptr<scalar_t>(),
                indices1.data_ptr<int64_t>(),
                indexptr1.data_ptr<int64_t>(),
                data2.data_ptr<scalar_t>(),
                indices2.data_ptr<int64_t>(),
                indexptr2.data_ptr<int64_t>(),
                out.data_ptr<scalar_t>(),
                N);
    });
    return out;
}


template <typename scalar_t>
static inline void sparse_square_norm_impl(
        scalar_t* data,
        int64_t* indptr,
        scalar_t* out_data,
        int64_t N) {
    int64_t i;
    int64_t i_start, i_end;
    scalar_t val_ij;
    // Assume squaring is desired, but need to fix this to accept multiple operations eventually
    for (i = 0; i < N; i++) {
        i_start = indptr[i];
        i_end = indptr[i+1];
        val_ij = 0.0;
        while (i_start < i_end) {
            val_ij += data[i_start]*data[i_start];
            i_start++;
        }
        out_data[i] = val_ij;
    }
}

at::Tensor sparse_square_norm_kernel(
        const at::Tensor &indexptr,
        const at::Tensor &data,
        at::Tensor &out) {
    ASSERT_IS_CPU(indexptr);
    AT_ASSERTM(indexptr.dim() == 1, "indexptr must be 1D");
    AT_ASSERTM(indexptr.stride(0) == 1, "indexptr must be memory-contiguous");
    int64_t N = indexptr.size(0) - 1;

    ASSERT_IS_CPU(data);
    AT_ASSERTM(data.dim() == 1, "data must be 1D");
    AT_ASSERTM(data.stride(0) == 1, "data must be memory-contiguous");

    ASSERT_IS_CPU(out);
    AT_ASSERTM(N == out.size(0), "Input shape mismatch");
    if (out.dim() >= 2) {
        AT_ASSERTM(out.size(1) == 1, "Output array must be 1D");
    }
    AT_ASSERTM(out.scalar_type() == data.scalar_type(), "Matrices A, B and out must be of the same type.");

    auto scalar_type = data.scalar_type();
    auto indexptr_data = indexptr.data_ptr<int64_t>();

    AT_DISPATCH_ALL_TYPES(scalar_type, "sparse_square_norm_impl", [&] {
        sparse_square_norm_impl<scalar_t>(
                data.data_ptr<scalar_t>(),
                indexptr_data,
                out.data_ptr<scalar_t>(),
                N);
    });
    return out;
}

template <typename scalar_t>
static inline void sparse_norm_impl(
        scalar_t* data,
        int64_t* indptr,
        scalar_t* out_data,
        int64_t N) {
    int64_t i;
    int64_t i_start, i_end;
    scalar_t val_ij;
    for (i = 0; i < N; i++) {
        i_start = indptr[i];
        i_end = indptr[i+1];
        val_ij = 0.0;
        while (i_start < i_end) {
            val_ij += data[i_start]*data[i_start];
            i_start++;
        }
        out_data[i] = sqrt(val_ij);
    }
}

at::Tensor sparse_norm_kernel(
        const at::Tensor &indexptr,
        const at::Tensor &data,
        at::Tensor &out) {
    ASSERT_IS_CPU(indexptr);
    AT_ASSERTM(indexptr.dim() == 1, "indexptr must be 1D");
    AT_ASSERTM(indexptr.stride(0) == 1, "indexptr must be memory-contiguous");
    int64_t N = indexptr.size(0) - 1;

    ASSERT_IS_CPU(data);
    AT_ASSERTM(data.dim() == 1, "data must be 1D");
    AT_ASSERTM(data.stride(0) == 1, "data must be memory-contiguous");

    ASSERT_IS_CPU(out);
    AT_ASSERTM(N == out.size(0), "Input shape mismatch");
    if (out.dim() >= 2) {
        AT_ASSERTM(out.size(1) == 1, "Output array must be 1D");
    }
    AT_ASSERTM(out.scalar_type() == data.scalar_type(), "Matrices A, B and out must be of the same type.");

    auto scalar_type = data.scalar_type();
    auto indexptr_data = indexptr.data_ptr<int64_t>();

    AT_DISPATCH_ALL_TYPES(scalar_type, "sparse_norm_impl", [&] {
        sparse_norm_impl<scalar_t>(
                data.data_ptr<scalar_t>(),
                indexptr_data,
                out.data_ptr<scalar_t>(),
                N);
    });
    return out;
}


} // namespace

TORCH_LIBRARY_IMPL(falkon, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::sparse_bdot"),
      TORCH_FN(sparse_bdot_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::sparse_square_norm"),
      TORCH_FN(sparse_square_norm_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("falkon::sparse_norm"),
      TORCH_FN(sparse_norm_kernel));
}

} // namespace ops
} // namespace falkon
