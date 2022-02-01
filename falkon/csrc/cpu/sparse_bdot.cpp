#include "sparse_bdot.h"
#include <torch/extension.h>

#define ASSERT_IS_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")


template <typename scalar_t>
static inline void calc_sparse_bdot(
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


torch::Tensor sparse_bdot_impl(
    const torch::Tensor &indexptr1,
    const torch::Tensor &indices1,
    const torch::Tensor &data1,
    const torch::Tensor &indexptr2,
    const torch::Tensor &indices2,
    const torch::Tensor &data2,
    torch::optional<torch::Tensor> out) {
    ASSERT_IS_CPU(indexptr1);
    ASSERT_IS_CPU(indices1);
    ASSERT_IS_CPU(data1);
    ASSERT_IS_CPU(indexptr2);
    ASSERT_IS_CPU(indices2);
    ASSERT_IS_CPU(data2);
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

    int64_t N = indexptr1.size(0) - 1;
    torch::Tensor out_;
    if (!out.has_value()) {
        out_ = torch::zeros({N, 1}, data1.options());
    } else {
        out_ = out.value();
        ASSERT_IS_CPU(out_);
        AT_ASSERTM(N == out_.size(0), "Input shape mismatch");
        if (out_.dim() >= 2) {
            AT_ASSERTM(out_.size(1) == 1, "Output array must be 1D");
        }
        AT_ASSERTM(out_.scalar_type() == data1.scalar_type(), "Matrices A, B and out must be of the same type.");
        // Fill tensor with zeros
        out_.fill_(0.0);
    }

    auto scalar_type = data1.scalar_type();

    AT_DISPATCH_ALL_TYPES(scalar_type, "calc_sparse_bdot", [&] {
        calc_sparse_bdot<scalar_t>(
                data1.data_ptr<scalar_t>(),
                indices1.data_ptr<int64_t>(),
                indexptr1.data_ptr<int64_t>(),
                data2.data_ptr<scalar_t>(),
                indices2.data_ptr<int64_t>(),
                indexptr2.data_ptr<int64_t>(),
                out_.data_ptr<scalar_t>(),
                N);
    });
    return out_;
}
