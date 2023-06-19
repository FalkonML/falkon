#include "cublas_bindings.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

void cublas_2d_copy_to_dev_async (
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& host_tensor,
        const int64_t lda,
        at::Tensor& dev_tensor,
        const int64_t ldb) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_2d_copy_to_dev_async", "")
                       .typed<decltype(cublas_2d_copy_to_dev_async)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        rows,
        cols,
        elemSize,
        host_tensor,
        lda,
        dev_tensor,
        ldb
    );
}
void cublas_2d_copy_to_dev (
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& host_tensor,
        const int64_t lda,
        at::Tensor& dev_tensor,
        const int64_t ldb) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_2d_copy_to_dev", "")
                       .typed<decltype(cublas_2d_copy_to_dev)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        rows,
        cols,
        elemSize,
        host_tensor,
        lda,
        dev_tensor,
        ldb
    );
}
void cublas_2d_copy_to_host_async(
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& dev_tensor,
        const int64_t lda,
        at::Tensor& host_tensor,
        const int64_t ldb) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_2d_copy_to_host_async", "")
                       .typed<decltype(cublas_2d_copy_to_host_async)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        rows,
        cols,
        elemSize,
        dev_tensor,
        lda,
        host_tensor,
        ldb
    );
}
void cublas_2d_copy_to_host(
        const int64_t rows,
        const int64_t cols,
        const int64_t elemSize,
        const at::Tensor& dev_tensor,
        const int64_t lda,
        at::Tensor& host_tensor,
        const int64_t ldb) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_2d_copy_to_host", "")
                       .typed<decltype(cublas_2d_copy_to_host)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        rows,
        cols,
        elemSize,
        dev_tensor,
        lda,
        host_tensor,
        ldb
    );
}

void cublas_trsm(
        const at::Tensor& A,
        at::Tensor& B,
        const at::Scalar& alpha,
        bool left,
        bool upper,
        bool transpose,
        bool unitriangular,
        int64_t m,
        int64_t n,
        int64_t lda,
        int64_t ldb) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_trsm", "")
                       .typed<decltype(cublas_trsm)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        A,
        B,
        alpha,
        left,
        upper,
        transpose,
        unitriangular,
        m,
        n,
        lda,
        ldb
    );
}

void cublas_trmm(
        const at::Tensor& A,
        const at::Tensor& B,
        at::Tensor& C,
        bool left,
        bool upper,
        bool transpose,
        bool unitriangular,
        const at::Scalar& alpha,
        int64_t m,
        int64_t n,
        int64_t lda,
        int64_t ldb,
        int64_t ldc) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_trmm", "")
                       .typed<decltype(cublas_trmm)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        A,
        B,
        C,
        left,
        upper,
        transpose,
        unitriangular,
        alpha,
        m,
        n,
        lda,
        ldb,
        ldc
    );
}


void cublas_gemm(
        const at::Tensor& A,
        int64_t lda,
        bool transa,
        const at::Tensor& B,
        int64_t ldb,
        bool transb,
        at::Tensor& C,
        int64_t ldc,
        int64_t m,
        int64_t n,
        int64_t k,
        const at::Scalar& alpha,
        const at::Scalar& beta) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_gemm", "")
                       .typed<decltype(cublas_gemm)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        A,
        lda,
        transa,
        B,
        ldb,
        transb,
        C,
        ldc,
        m,
        n,
        k,
        alpha,
        beta
    );
}


void cublas_syrk(
        const at::Tensor& A,
        int64_t lda,
        at::Tensor& C,
        int64_t ldc,
        const at::Scalar& alpha,
        const at::Scalar& beta,
        bool upper,
        bool transpose,
        int64_t n,
        int64_t k) {
    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::cublas_syrk", "")
                       .typed<decltype(cublas_syrk)>();
    at::AutoDispatchBelowAutograd guard;
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return op.call(
        A,
        lda,
        C,
        ldc,
        alpha,
        beta,
        upper,
        transpose,
        n,
        k
    );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_2d_copy_to_dev_async(int rows, int cols, int elemSize, Tensor host_tensor, int lda, Tensor (a!) dev_tensor, int ldb) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_2d_copy_to_dev(int rows, int cols, int elemSize, Tensor host_tensor, int lda, Tensor (a!) dev_tensor, int ldb) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_2d_copy_to_host_async(int rows, int cols, int elemSize, Tensor dev_tensor, int lda, Tensor (a!) host_tensor, int ldb) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_2d_copy_to_host(int rows, int cols, int elemSize, Tensor dev_tensor, int lda, Tensor (a!) host_tensor, int ldb) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_trsm(Tensor A, Tensor (a!) B, Scalar alpha, bool left, bool upper, bool transpose, bool unitriangular, int m, int n, int lda, int ldb) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_trmm(Tensor A, Tensor B, Tensor (a!) C, bool left, bool upper, bool transpose, bool unitriangular, Scalar alpha, int m, int n, int lda, int ldb, int ldc) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_gemm(Tensor A, int lda, bool transa, Tensor B, int ldb, bool transb, Tensor (a!) C, int ldc, int m, int n, int k, Scalar alpha, Scalar beta) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::cublas_syrk(Tensor A, int lda, Tensor (a!) C, int ldc, Scalar alpha, Scalar beta, bool upper, bool transpose, int n, int k) -> ()"));
}

} // namespace ops
} // namespace falkon
