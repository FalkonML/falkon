#include <ATen/Parallel.h>
#include<ATen/ParallelOpenMP.h>
//#include<ATen/ParallelNative.h>
#include <torch/extension.h>

#define INDEX(A, i, j, si, sj) ( A[i * si + j * sj] )
#define ASSERT_IS_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")


template <typename scalar_t>
static void run_parallel(scalar_t* A_data, scalar_t* B_data, scalar_t* out_data,
                         int64_t* A_indptr, int64_t* B_indptr, int64_t* A_index, int64_t* B_index,
                         int64_t si, int64_t sj, int64_t N, int64_t M) {
    if (si > 1) {
        torch::parallel_for(0, N, 2048, [&](int64_t start, int64_t end) {
            int64_t i, j, i_start, i_end, j_start, j_end;
            int64_t row_val, col_val;
            scalar_t val_ij;
            for (i = start; i < end; i++) {
                i_start = A_indptr[i];
                i_end = A_indptr[i+1];
                if (A_indptr[i] == A_indptr[i+1]) {
                    continue;
                }
                auto out_row = &out_data[i*si];
                for (j = 0; j < M; j++) {
                    j_start = B_indptr[j];
                    j_end = B_indptr[j+1];
                    val_ij = out_row[j];

                    while (j_start < j_end && i_start < i_end) {
                        row_val = A_index[i_start];
                        col_val = B_index[j_start];
                        if (row_val == col_val) {
                            val_ij += A_data[i_start] * B_data[j_start];
                            j_start++;
                            i_start++;
                        } else if (row_val < col_val) {
                            i_start++;
                        } else {
                            j_start++;
                        }
                    }
                    out_data[j] = val_ij;
                    //INDEX(out_data, i, j, si, sj) = val_ij;
                }
            }
        });
    } else {
        at::parallel_for(0, N, 2048, [&](int64_t begin, int64_t end) {
            int64_t i, j, i_start, i_end, j_start, j_end;
            int64_t row_val, col_val;
            scalar_t val_ij;
            for (j = begin; j < end; j++) {
                if (B_indptr[j] == B_indptr[j+1]) {
                    continue;
                }
                for (i = 0; i < N; i++) {
                    i_start = A_indptr[i];
                    i_end = A_indptr[i+1];
                    j_start = B_indptr[j];
                    j_end = B_indptr[j+1];
                    val_ij = INDEX(out_data, i, j, si, sj);

                    while (j_start < j_end && i_start < i_end) {
                        row_val = A_index[i_start];
                        col_val = B_index[j_start];
                        if (row_val == col_val) {
                            val_ij += A_data[i_start] * B_data[j_start];
                            j_start++;
                            i_start++;
                        } else if (row_val < col_val) {
                            i_start++;
                        } else {
                            j_start++;
                        }
                    }
                    INDEX(out_data, i, j, si, sj) = val_ij;
                }
            }
        });
    }
}


torch::Tensor addspmm(torch::Tensor rowptrA, torch::Tensor colA, torch::Tensor valA,
                      torch::Tensor colptrB, torch::Tensor rowB, torch::Tensor valB,
                      torch::optional<torch::Tensor> out=torch::nullopt) {
    ASSERT_IS_CPU(rowptrA);
    AT_ASSERTM(rowptrA.dim() == 1, "rowptrA must be 1D");
    const int64_t N = rowptrA.size(0) - 1;

    ASSERT_IS_CPU(colA);
    AT_ASSERTM(colA.dim() == 1, "colA must be 1D");

    ASSERT_IS_CPU(valA);
    AT_ASSERTM(valA.dim() == 1, "valA must be 1D");
    AT_ASSERTM(valA.size(0) == colA.size(0), "Values and column-data must be of the same size for matrix A");

    ASSERT_IS_CPU(colptrB);
    AT_ASSERTM(colptrB.dim() == 1, "rowptrB must be 1D");
    const int64_t M = colptrB.size(0) - 1;

    ASSERT_IS_CPU(rowB);
    AT_ASSERTM(rowB.dim() == 1, "rowB must be 1D");

    ASSERT_IS_CPU(valB);
    AT_ASSERTM(valB.dim() == 1, "valB must be 1D");
    AT_ASSERTM(valB.size(0) == rowB.size(0), "Values and column-data must be of the same size for matrix B");
    AT_ASSERTM(valB.scalar_type() == valA.scalar_type(), "Matrices A, B and out must be of the same type.");

    torch::Tensor out_;
    if (!out.has_value()) {
        out_ = torch::zeros({N, M}, valA.options());
    } else {
        out_ = out.value();
        ASSERT_IS_CPU(out_);
        AT_ASSERTM(N == out_.size(0), "Input shape mismatch");
        AT_ASSERTM(M == out_.size(1), "Input shape mismatch");
        AT_ASSERTM(out_.scalar_type() == valA.scalar_type(), "Matrices A, B and out must be of the same type.");
    }

    const auto scalar_type = valA.scalar_type();
    const auto si = out_.stride(0);
    const auto sj = out_.stride(1);

    const auto rowptrA_data = rowptrA.data_ptr<int64_t>();
    const auto colA_data = colA.data_ptr<int64_t>();
    const auto colptrB_data = colptrB.data_ptr<int64_t>();
    const auto rowB_data = rowB.data_ptr<int64_t>();

    AT_DISPATCH_ALL_TYPES(scalar_type, "call", [&] {
        run_parallel<scalar_t>(
                valA.data_ptr<scalar_t>(),
                valB.data_ptr<scalar_t>(),
                out_.data_ptr<scalar_t>(),
                rowptrA_data,
                colptrB_data,
                colA_data,
                rowB_data,
                si, sj, N, M);
    });
    return out_;
}
