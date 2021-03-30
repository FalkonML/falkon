#include <torch/extension.h>
#include <math.h>

#define ASSERT_IS_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")

template <typename scalar_t>
static inline void calc_norm_sq(
    scalar_t* data, int64_t* indptr,
    scalar_t* out_data, int64_t N) {

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


torch::Tensor norm_sq(torch::Tensor indexptr, torch::Tensor data,
                      torch::optional<torch::Tensor> out=torch::nullopt) {
    ASSERT_IS_CPU(indexptr);
    AT_ASSERTM(indexptr.dim() == 1, "indexptr must be 1D");
    AT_ASSERTM(indexptr.stride(0) == 1, "indexptr must be memory-contiguous");
    int64_t N = indexptr.size(0) - 1;

    ASSERT_IS_CPU(data);
    AT_ASSERTM(data.dim() == 1, "data must be 1D");
    AT_ASSERTM(data.stride(0) == 1, "data must be memory-contiguous");

    torch::Tensor out_;
    if (!out.has_value()) {
        out_ = torch::zeros({N, 1}, data.options());
    } else {
        out_ = out.value();
        ASSERT_IS_CPU(out_);
        AT_ASSERTM(N == out_.size(0), "Input shape mismatch");
        if (out_.dim() >= 2) {
            AT_ASSERTM(out_.size(1) == 1, "Output array must be 1D");
        }
        AT_ASSERTM(out_.scalar_type() == data.scalar_type(), "Matrices A, B and out must be of the same type.");
    }

    auto scalar_type = data.scalar_type();
    auto indexptr_data = indexptr.data_ptr<int64_t>();

    AT_DISPATCH_ALL_TYPES(scalar_type, "calc_norm_sq", [&] {
        calc_norm_sq<scalar_t>(
                data.data_ptr<scalar_t>(),
                indexptr_data,
                out_.data_ptr<scalar_t>(),
                N);
    });
    return out_;
}

template <typename scalar_t>
static inline void calc_norm(
    scalar_t* data, int64_t* indptr,
    scalar_t* out_data, int64_t N) {

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
        out_data[i] = sqrt(val_ij);
    }
}


torch::Tensor norm(torch::Tensor indexptr, torch::Tensor data,
                   torch::optional<torch::Tensor> out=torch::nullopt) {
    ASSERT_IS_CPU(indexptr);
    AT_ASSERTM(indexptr.dim() == 1, "indexptr must be 1D");
    AT_ASSERTM(indexptr.stride(0) == 1, "indexptr must be memory-contiguous");
    int64_t N = indexptr.size(0) - 1;

    ASSERT_IS_CPU(data);
    AT_ASSERTM(data.dim() == 1, "data must be 1D");
    AT_ASSERTM(data.stride(0) == 1, "data must be memory-contiguous");

    torch::Tensor out_;
    if (!out.has_value()) {
        out_ = torch::zeros({N, 1}, data.options());
    } else {
        out_ = out.value();
        ASSERT_IS_CPU(out_);
        AT_ASSERTM(N == out_.size(0), "Input shape mismatch");
        if (out_.dim() >= 2) {
            AT_ASSERTM(out_.size(1) == 1, "Output array must be 1D");
        }
        AT_ASSERTM(out_.scalar_type() == data.scalar_type(), "Matrices A, B and out must be of the same type.");
    }

    auto scalar_type = data.scalar_type();
    auto indexptr_data = indexptr.data_ptr<int64_t>();

    AT_DISPATCH_ALL_TYPES(scalar_type, "calc_norm", [&] {
        calc_norm<scalar_t>(
                data.data_ptr<scalar_t>(),
                indexptr_data,
                out_.data_ptr<scalar_t>(),
                N);
    });
    return out_;
}
