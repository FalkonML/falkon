#include "trtri.h"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cublas_v2.h>

#include "utils.cuh"

#define BLOCK_SIZE 32


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
__global__ void
dev_trtri_L_registers_Nfix(const int n, scalar_t* A, const int lda)
{
  scalar_t rA[TX], s, a;
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++) {
      rA[i] = __ldg(&(A[ threadIdx.x + i * lda ]));
  }
  //perform inverting on registers
  #pragma unroll
  for(int j = TX - 1; j >= 0; j--) {
    s = 0.0;
    #pragma unroll
    for(int i = 0; i < TX; i++) {
      a = shfl(rA[j], i, TX);
      if(j < i && i <= threadIdx.x) {
        s = FMA( rA[i], a, s);
      }
    }
    a = shfl(rA[j], j, TX);
    if(threadIdx.x == j) {
      rA[j] = 1.0 / a;
    }
    else if(threadIdx.x > j) {
      rA[j] = -s / a;
    }
  }
  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++) {
    if(i <= threadIdx.x) {
      A[ threadIdx.x + i * lda ] = rA[ i ];
    }
  }
}


template<typename scalar_t, int TX>
__global__ void
dev_trtri_U_registers_Nfix(const int n, scalar_t* A, const int lda)
{
  scalar_t rA[TX], s, a;
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++) {
      rA[i] = __ldg(&(A[ threadIdx.x + i * lda ]));
  }
  //perform inverting on registers
  #pragma unroll
  for(int j = 0; j < TX; j++) {
    s = 0.0;
    #pragma unroll
    for(int i = TX - 1; i >= 0; i--) {
      a = shfl(rA[j], i, TX);
      if(i < j && i >= threadIdx.x) {
        s = FMA( rA[i], a, s);
      }
    }
    a = shfl(rA[j], j, TX);
    if(threadIdx.x == j) {
      rA[j] = 1.0 / a;
    }
    else if(threadIdx.x < j) {
      rA[j] = -s / a;
    }
  }
  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++) {
    if(i >= threadIdx.x) {
      A[ threadIdx.x + i * lda ] = rA[ i ];
    }
  }
}


/* TRSM (cuBLAS) */
template<typename scalar_t>
inline void trsm(const cublasHandle_t cublas_handle, const cublasFillMode_t uplo, const cublasDiagType_t diag,
          const int m, const int n, const scalar_t *A, int lda, scalar_t *B, int ldb)
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
          double *B,
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
          float *B,
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
    cublasHandle_t cublas_handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasDiagType_t diag, int m, int n, scalar_t alpha,
    const scalar_t *A, int lda, scalar_t *B, int ldb)
{ throw std::invalid_argument("scalar_t"); }
template<>
inline void trmm<double>(
    cublasHandle_t cublas_handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int m,
    int n,
    double alpha,
    const double *A, int lda,
    double *B,  int ldb)
{
    TORCH_CUDABLAS_CHECK(cublasDtrmm(
        /*handle=*/cublas_handle, /*side=*/side, /*fill*/uplo, /*operation=*/CUBLAS_OP_N,
        /*diag=*/diag, /*m=*/m, /*n=*/n, /*alpha=*/&alpha, /*A=*/A, /*lda=*/lda, /*B=*/B, /*ldb=*/ldb,
        /*C=*/B, /*ldc=*/ldb
    ));
}
template<>
inline void trmm<float>(
    cublasHandle_t cublas_handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int m,
    int n,
    float alpha,
    const float *A, int lda,
    float *B,  int ldb)
{
    TORCH_CUDABLAS_CHECK(cublasStrmm(
        /*handle=*/cublas_handle, /*side=*/side, /*fill*/uplo, /*operation=*/CUBLAS_OP_N,
        /*diag=*/diag, /*m=*/m, /*n=*/n, /*alpha=*/&alpha, /*A=*/A, /*lda=*/lda, /*B=*/B, /*ldb=*/ldb,
        /*C=*/B, /*ldc=*/ldb
    ));
}



template<typename scalar_t, int TX>
__global__ void
dev_trtridiag_U_registers_Nfix(const int n, scalar_t* A, const int lda) 
{
    // Grid is 1D, each block is 1D
    // Each element of the grid encodes for one diagonal block of size TX
    // go left to right
    int block = blockIdx.x * blockDim.x;
    int mat_offset = block + block * lda;
    scalar_t* oA = A + mat_offset;
    dev_trtri_U_registers_Nfix<scalar_t, TX>(TX, oA, lda);
    /*
    // Grid is 1D, each block is 1D
    // Each element of the grid encodes for one diagonal block of size TX
    // go left to right
    int block = blockIdx.x * blockDim.x;
    int mat_offset = block + block * lda;
    scalar_t* oA = A + mat_offset;

    scalar_t rA[TX], s, a;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++) {
        rA[i] = __ldg(&(oA[ threadIdx.x + i * lda ]));
    }
    //perform inverting on registers
    #pragma unroll
    for(int j = 0; j < TX; j++) {
        s = 0.0;
        #pragma unroll
        for(int i = TX - 1; i >= 0; i--) {
            a = shfl(rA[j], i, TX);
            if(i < j && i >= threadIdx.x) {
                s = FMA( rA[i], a, s);
            }
        }
        a = shfl(rA[j], j, TX);
        if(threadIdx.x == j) {
            rA[j] = 1.0 / a;
        }
        else if(threadIdx.x < j) {
            rA[j] = -s / a;
        }
    }
    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++) {
        if(i >= threadIdx.x) {
            oA[ threadIdx.x + i * lda ] = rA[ i ];
        }
    }*/
}


template<typename scalar_t>
void dev_trtridiag_upper(int n, scalar_t *data, int lda) {
    if (n % BLOCK_SIZE != 0) {
        AT_ERROR("Block size incorrect");
    }
    const dim3 dimGrid(n / BLOCK_SIZE, 1);
    const dim3 dimBlock(BLOCK_SIZE, 1);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dev_trtridiag_U_registers_Nfix<scalar_t, BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream.stream()>>>(n, data, lda);
}

template<typename scalar_t>
void dev_trtri_upper(int m, scalar_t *data, int lda) {
    const dim3 dimGrid(1, 1);
    const dim3 dimBlock(BLOCK_SIZE, 1);

    if (m != BLOCK_SIZE) {
        // Error
        AT_ERROR("Block size incorrect");
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dev_trtri_U_registers_Nfix<scalar_t, BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream.stream()>>>(m, data, lda);
}


template<typename scalar_t>
void dev_trtri_lower(int m, scalar_t *data, int lda) {
    const dim3 dimGrid(1, 1);
    const dim3 dimBlock(BLOCK_SIZE, 1);

    if (m != BLOCK_SIZE) {
        // Error
        AT_ERROR("Block size incorrect");
    }
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    dev_trtri_L_registers_Nfix<scalar_t, BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream.stream()>>>(m, data, lda);
}


template<typename scalar_t>
void _trtri_upper(int n, scalar_t *data_ptr, int nb, int lda, cublasHandle_t cublas_handle) {

    dev_trtridiag_upper<scalar_t>(/*n=*/n, /*A=*/data_ptr, /*lda=*/lda);

    int jb;
    for (int j=0; j < n; j += nb) {
        jb = min(nb, n-j); 
        
        if (j > 0) {
            // First j rows of block-column (j to j+jb)
            trmm<scalar_t>(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT,
                           /*m=*/j, /*n=*/jb, /*alpha=*/(scalar_t)1.0, /*A=*/(data_ptr), /*lda=*/lda,
                           /*B=*/(data_ptr + j * lda), /*ldb=*/lda);
            trmm<scalar_t>(cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT,
                           /*m=*/j, /*n=*/jb, /*alpha=*/(scalar_t)-1.0, /*A=*/(data_ptr + j + j * lda), /*lda=*/lda,
                           /*B=*/(data_ptr + j * lda), /*ldb=*/lda);

            //trsm<scalar_t>(cublas_handle,  CUBLAS_FILL_MODE_UPPER, CUBLAS_DIAG_NON_UNIT,
            //               /*m=*/j, /*n=*/jb, /*A=*/(data_ptr + j + j * lda), /*lda=*/lda,
            //               /*B=*/(data_ptr + j * lda), /*ldb=*/lda);
        }
        // Last jb rows (triangular part) of block-column (j to j+jb)
        //dev_trtri_upper<scalar_t>(/*m=*/jb, /*A=*/(data_ptr + j + j * lda), /*lda=*/lda);
    }
}


template<typename scalar_t>
void _trtri_lower(int n, scalar_t *data_ptr, int nb, int lda, cublasHandle_t cublas_handle) {
    int nn = ((n - 1)/nb)*nb;
    int jb;
    for (int j = nn; j >= 0; j -= nb) {
        jb = min(nb, n-j);
        if (j < nn) {
            // Compute rows j+jb to n of the current block-column (j to j+jb)
            trmm<scalar_t>(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT,
                           /*m=*/n-(j+jb), /*n=*/jb, /*alpha=*/(scalar_t)1.0, /*A=*/data_ptr + j + jb + (j + jb) * lda, /*lda=*/lda,
                           /*B=*/data_ptr + j + jb + j * lda, /*ldb=*/lda);
            trsm<scalar_t>(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT,
                           /*m=*/n-(j+jb), /*n=*/jb, /*A=*/data_ptr + j + j * lda, /*lda=*/lda,
                           /*B=*/data_ptr + j + jb + j * lda, /*ldb=*/lda);
        }
        // Inverse of the diagonal block: rows j to j + jb of block column j to j + jb.
        dev_trtri_lower<scalar_t>(/*m=*/jb, /*A=*/(data_ptr + j + j * lda), /*lda=*/lda);
    }
}


torch::Tensor trtri_cuda(torch::Tensor &A, const bool lower, const bool unitdiag) {
    CHECK_CUDA(A);
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square 2D matrix.");

    bool blower = is_fortran_contig(A) ? lower : !lower;

    const auto scalar_type = A.scalar_type();
    const int size = A.size(0);
    const int lda = is_fortran_contig(A) ? A.stride(1) : A.stride(0);
    const auto cublas_handle = at::cuda::getCurrentCUDABlasHandle();

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch_trtri_cuda", [&] {
        at::DeviceGuard g(A.device());
        if (blower) {
            _trtri_lower<scalar_t>(size, A.data_ptr<scalar_t>(), BLOCK_SIZE, lda, cublas_handle);
        } else {
            _trtri_upper<scalar_t>(size, A.data_ptr<scalar_t>(), BLOCK_SIZE, lda, cublas_handle);
        }
    });
    return A;
}

