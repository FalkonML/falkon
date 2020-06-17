/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xlauum_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XLAUUM_BATCH_KERNELS_H__
#define __XLAUUM_BATCH_KERNELS_H__


//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
//Naming convention <dev/kernel>_<KernelName>_<Non/Uniform>_<Right/Left><Lower/Upper><Non/Transpose><Non/Diag>_<variants>
//==============================================================================================
#ifndef TARGET_SM
  #error "TARGET_SM is not defined"
#elif (TARGET_SM >= 30)


//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_lauum_U_reg_shared_Nfix(int n, T* A, int lda)
{
  const int TX1 = TX + 2;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;

  T rA[TX], s, a, zero = make_zero<T>();

  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX1 ] = __ldg(&(A[ tx + i * lda ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rA[ i ] = sdata[ i + tx * TX1 ];//TODO handle bank conflicts
  }

  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = zero;
    #pragma unroll
    for(int i = 0; i < TX; i++){
      a = shfl(rA[i], j, TX);
      if(j <= i && j >= tx)
        s = FMA( rA[i], a, s);
    }
    rA[j] = s;
  }

  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX1 ] = rA[ i ];
  }
  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      A[ tx + i * lda ] = sdata[ tx + i*TX1 ];
  }
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_lauum_U_reg_shared_Nfix( const int n, int batchCount,
                                T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n != TX){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_lauum_U_reg_shared_Nfix<T, TX>(n, A, lda);
}


//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_lauum_U_registers_Nfix(int n, T* A, int lda)
{
  T rA[TX], s, a, zero = make_zero<T>();

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rA[ i ] = __ldg(&(A[ i + tx * lda ]));
  }

  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = zero;
    #pragma unroll
    for(int i = 0; i < TX; i++){
      a = shfl(rA[i], j, TX);
      if(j <= i && j >= tx)
        s = FMA( rA[i], a, s);
    }
    rA[j] = s;
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx <= i)
      A[ i + tx * lda ] = rA[ i ];
  }
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_lauum_U_registers_Nfix(const int n, int batchCount,
                              T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                              int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n != TX){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_lauum_U_registers_Nfix<T, TX>(n, A, lda);
}

//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_lauum_U_registers_Nvar(int n, T* A, int lda)
{
  T rA[TX], s, a, zero = make_zero<T>();

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    if(tx < n && i < n)
      rA[ i ] = __ldg(&(A[ i + tx * lda ]));
  }

  #pragma unroll
  for(int j = 0; j < TX; j++){
    if(j < n){
      s = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        a = shfl(rA[i], j, TX);
        if(j <= i && i < n && j >= tx)
          s = FMA( rA[i], a, s);
      }
      rA[j] = s;
    }
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx < n && tx <= i && i < n)
      A[ i + tx * lda ] = rA[ i ];
  }
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_lauum_U_registers_Nvar(const int n, int batchCount,
                              T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                              int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n > TX){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_lauum_U_registers_Nvar<T, TX>(n, A, lda);
}

//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch LAUUM"
#endif

#endif //__XLAUUM_BATCH_KERNELS_H__
