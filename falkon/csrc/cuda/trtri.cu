#include "trtri.cuh"

#include <stdio.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cublas_v2.h>

#include "utils.cuh"

#define BLOCK_SIZE 4
#define TORCH_CUDABLAS_CHECK(EXPR)                                      \
  do {                                                          \
    cublasStatus_t __err = EXPR;                              \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 \
                "CuBLAS error: ",                               \
                cublasGetErrorString(__err),                    \
                " when calling `" #EXPR "`");                   \
  } while (0)

const char* cublasGetErrorString(cublasStatus_t error) {
  if (error == CUBLAS_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == CUBLAS_STATUS_ALLOC_FAILED) {
    return "CUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == CUBLAS_STATUS_INVALID_VALUE) {
    return "CUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == CUBLAS_STATUS_MAPPING_ERROR) {
    return "CUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

/* cu* library data and functions */
static constexpr double const oned = 1.0;
static constexpr double const moned = -1.0;
static constexpr float const onef = 1.0;
static constexpr float const monef = -1.0;
__device__ static __inline__ float FMA(float a, float b, float c){return fmaf(a,b,c);}
__device__ static __inline__ double FMA(double a, double b, double c){return fma(a,b,c);}
__device__ __inline__ float shfl(float x, int lane, int ws = 32)
{
  return __shfl_sync(0xFFFFFFFF, x, lane, ws);
}
__device__ __inline__ double shfl(double x, int lane, int ws = 32)
{
  // Split the double into 2 32b registers.
  int lo = __double2loint(x), hi = __double2hiint(x);
  // Shuffle the two 32b registers.
  lo = __shfl_sync(0xFFFFFFFF, lo, lane, ws);
  hi = __shfl_sync(0xFFFFFFFF, hi, lane, ws);
  // Recreate the 64b number.
  return __hiloint2double(hi,lo);
}

template<typename scalar_t, int TX>
__device__ inline void
dev_trtri_U_registers_Nfix(int n, scalar_t* A, int lda)
{
  scalar_t rA[TX], s, a;
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    //if(tx >= i)
      rA[i] = __ldg(&(A[ threadIdx.x + i * lda ]));
  }

  //perform inverting on registers
  #pragma unroll
  for(int j = TX-1; j >= 0; j--)
  {
    s = 0.0;
    #pragma unroll
    for(int i = 0; i < TX; i++){
      a = shfl(rA[j], i, TX);
      if(j < i && i <= tx)
        s = FMA( rA[i], a, s);
    }
    a = shfl(rA[j], j, TX);
    if(tx == j)
      rA[j] = 1.0 / a;
    else if(tx > j)
      rA[j] = -s / a;
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      A[ tx + i * lda ] = rA[ i ];
  }
}


/* TRSM (cuBLAS) */
template<typename scalar_t>
inline void trsm(const cublasHandle_t cublas_handle, const cublasFillMode_t uplo, const cublasDiagType_t diag,
          const int m, const int n, const scalar_t *A, int lda, const scalar_t *B, int ldb)
{ throw std::invalid_argument("scalar_t"); }
template<>
inline void trsm<double>(
          const cublasHandle_t cublas_handle,
          const cublasFillMode_t uplo,
          const cublasDiagType_t diag,
          const int m,
          const int n,
          const double *A,
          int lda,
          const double *B,
          int ldb)
{
    TORCH_CUDABLAS_CHECK(cublasDtrsm(
        /*handle=*/cublas_handle, /*side=*/CUBLAS_SIDE_RIGHT, /*uplo=*/uplo,
        /*trans=*/CUBLAS_OP_N, /*diag=*/diag,
        /*m=*/m, /*n=*/n, /*alpha=*/&moned,
        /*A=*/A, /*lda=*/lda, /*B=*/B, /*ldb=*/ldb
    ));
}
template<>
inline void trsm<float>(
          const cublasHandle_t cublas_handle,
          const cublasFillMode_t uplo,
          const cublasDiagType_t diag,
          const int m,
          const int n,
          const float *A,
          int lda,
          const float *B,
          int ldb)
{
    TORCH_CUDABLAS_CHECK(cublasStrsm(
        /*handle=*/cublas_handle, /*side=*/CUBLAS_SIDE_RIGHT, /*uplo=*/uplo,
        /*trans=*/CUBLAS_OP_N, /*diag=*/diag,
        /*m=*/m, /*n=*/n, /*alpha=*/&monef,
        /*A=*/A, /*lda=*/lda, /*B=*/B, /*ldb=*/ldb
    ));
}

/* TRMM (cuBLAS) */
template<typename scalar_t>
inline void trmm(
    const cublasHandle_t cublas_handle, const cublasSideMode_t side, const cublasFillMode_t uplo,
    const cublasDiagType_t diag, const int m, const int n,
    const scalar_t *A, int lda, scalar_t *B, int ldb)
{ throw std::invalid_argument("scalar_t"); }
template<>
inline void trmm<double>(
    const cublasHandle_t cublas_handle,
    const cublasSideMode_t side,
    const cublasFillMode_t uplo,
    const cublasDiagType_t diag,
    const int m,
    const int n,
    const double *A, int lda,
    double *B,  int ldb)
{
    TORCH_CUDABLAS_CHECK(cublasDtrmm(
        /*handle=*/cublas_handle, /*side=*/side, /*fill*/uplo, /*operation*/=CUBLAS_OP_N,
        /*diag=*/diag, /*m=*/m, /*n=*/n, /*alpha=*/&oned, /*A=*/A, /*lda=*/lda, /*B=*/B, /*ldb=*/ldb,
        /*C=*/B, /*ldc=*/ldc
    ));
}
template<>
inline void trmm<float>(
    const cublasHandle_t cublas_handle,
    const cublasSideMode_t side,
    const cublasFillMode_t uplo,
    const cublasDiagType_t diag,
    const int m,
    const int n,
    const float *A, int lda,
    float *B,  int ldb)
{
    TORCH_CUDABLAS_CHECK(cublasDtrmm(
        /*handle=*/cublas_handle, /*side=*/side, /*fill*/uplo, /*operation*/=CUBLAS_OP_N,
        /*diag=*/diag, /*m=*/m, /*n=*/n, /*alpha=*/&onef, /*A=*/A, /*lda=*/lda, /*B=*/B, /*ldb=*/ldb,
        /*C=*/B, /*ldc=*/ldc
    ));
}


template<typename scalar_t>
void dev_trtri_upper(int m, scalar_t *data, int lda) {

    // Setup CUDA grid dimensions:
    // grid is 1D, so that we can only consider triangularly-appropriate tiles
    // blocks are 2D, with a fixed block size
    const dim3 dimGrid(1, 1);
    const dim3 dimBlock(BLOCK_SIZE, 1);

    if (m != BLOCK_SIZE) {
        // Error
        TORCH_ERROR("Block size incorrect");
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dev_trtri_U_registers_Nfix<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(m, data, lda);
}


template<typename scalar_t>
void _trtri_upper(const int n, torch::Tensor &mat, const int nb, const int lda, cublasHandle_t cublas_handle) {
    scalar_t *data_ptr = mat.data_ptr<scalar_t>();
    for (int j=0; j < n; j += nb) {
        jb = min(nb, n-j);
        if (j > 0) {
            // First j rows of block-column j-j+jb
            trmm<scalar_t>(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT,
                           /*m=*/j, /*n=*/jb, /*A=*/(void *)(data_ptr), /*lda=*/lda,
                           /*B=*/(void *)(data_ptr + j * lda), /*ldb=*/ldb);

            trsm<scalar_t>(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT,
                           /*m=*/j, /*n=*/jb, /*A=*/(void *)(data_ptr + j + j * lda), /*lda=*/lda,
                           /*B=*/(void *)(data_ptr + j * lda), /*ldb=*/ldb);
        }
        // Last jb rows (triangular part) of block-column j-j+jb
        dev_trtri_upper<scalar_t>(/*m=*/jb, /*A=*/(void *)(data_ptr + j + j * lda), /*lda=*/lda);
    }
}


torch::Tensor trtri_cuda(torch::Tensor &A, const bool lower, const bool unitdiag) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    if (lower) {
        TORCH_ERROR("Must be upper");
    }
    if (A.stride(0) != 1) {
        TORCH_ERROR("A must be f-contig");
    }
    const auto scalar_type = A.scalar_type();
    const int lda = A.stride(1);
    const auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch_lauum_cuda", [&] {
        at::DeviceGuard g(A.device());
        _trtri_upper<scalar_t>(A.size(0), A, BLOCK_SIZE, lda, cublas_handle);
    });
    return A;
}

