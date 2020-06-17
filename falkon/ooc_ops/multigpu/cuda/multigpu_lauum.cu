/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

       See [zcds]gemm_fermi.cu for description of related files.
*/

#ifndef GEMM_TEMPLATE_DEVICE_CUH
#define GEMM_TEMPLATE_DEVICE_CUH



/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int THR_M, const int THR_N>
static __device__
void gemm_template_device_tn(
    int M, int N, int K,
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta )
{
    int idx = threadIdx.x;  // thread's m dimension
    int idy = threadIdx.y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = blockIdx.x;   // block's m dimension
    int bly = blockIdx.y;   // block's n dimension

    __shared__ T sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ T sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T rA[THR_M];
    T rB[THR_N];

    // Registers for the dev->shmem copy
    T ra[BLK_M/DIM_YA][BLK_K/DIM_XA];
    T rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

    // bound is the correction to offs_d in order to not get out of memory bound
    // so bound could be negative value since offs_d could be out of bound
    const T *offs_dA = A + blx*BLK_M*LDA + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(M-1) + K) - ( blx*BLK_M*LDA + idyA*LDA + idxA ) -1;

    const T *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = 0.0;

    // Load A dev->shmem
    #pragma unroll
    for (n = 0; n < BLK_M; n += DIM_YA)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XA) {
            sA[m+idxA][n+idyA] = offs_dA[min(n*LDA+m, boundA)];
            // sA[m+idxA][n+idyA] = fetch(A, m, n, boundA);
        }

    // Load B dev->shmem
    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB)
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB) {
            sB[n+idyB][m+idxB] = offs_dB[min(n*LDB+m, boundB)];
            // sB[n+idyB][m+idxB] = fetch(B, m, n, boundB);
        }

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K;
        boundA  -= BLK_K;

        offs_dB += BLK_K;
        boundB  -= BLK_K;

        // Load A dev->regs
        #pragma unroll
        for (n = 0; n < BLK_M/DIM_YA; n++){
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XA; m++) {
                ra[n][m] = offs_dA[min(n*DIM_YA*LDA+m*DIM_XA, boundA)];
                // ra[n][m] = fetch(A, m*DIM_XA, n*DIM_YA, boundA);
            }
        }
        // Load B dev->regs
        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++){
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++){
                rb[n][m] = offs_dB[min(n*DIM_YB*LDB+m*DIM_XB, boundB)];
                //rb[n][m] = fetch(B, m*DIM_XB, n*DIM_YB, boundB);
            }
        }
        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    rC[n][m] += rA[m] * rB[n];
                }
            }
        }

        __syncthreads();

        // Load A regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_M/DIM_YA; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XA; m++)
                sA[m*DIM_XA+idxA][n*DIM_YA+idyA] = ra[n][m];

        // Load B regs->shmem
        #pragma unroll
        for (n = 0; n < BLK_N/DIM_YB; n++)
            #pragma unroll
            for (m = 0; m < BLK_K/DIM_XB; m++)
                sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                rC[n][m] += rA[m] * rB[n];
            }
        }
    }

    // Store C regs->dev
    if( beta == 0.0 ){
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    int offsC = coord_dCn*LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = alpha * regC;
                }
            }
        }
    }else{
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                int coord_dCm = blx*BLK_M + m*DIM_X + idx;
                if (coord_dCm < M && coord_dCn < N) {
                    int offsC = coord_dCn*LDC + coord_dCm;

                    T &regC = rC[n][m];
                    T &memC = C[offsC];

                    memC = alpha * regC + beta * memC;
                }
            }
        }
    }
}


template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB>
static __global__
void gemm_template_batched_tn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    const int batchid = blockIdx.z;

    gemm_template_device_tn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y)>
        ( M, N, K,
          Aarray[batchid] + LDA *  coffA + roffA, LDA,
          Barray[batchid] + LDB *  coffB + roffB, LDB,
          Carray[batchid] + LDC *  coffC + roffC, LDC,
          alpha, beta );
}


template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB>
void gemm_template_batched_tn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t ncalls = magma_ceildiv(batchCount, 65536);
    magma_int_t batchCount_percall = batchCount/ncalls;

    for(magma_int_t batch_starting_id=0; batch_starting_id<batchCount; batch_starting_id+=batchCount_percall)
    {
        magma_int_t this_batchCount = min(batchCount_percall, batchCount-batch_starting_id);

        dim3 dimBlock(DIM_X, DIM_Y);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), this_batchCount );
        gemm_template_batched_tn_kernel
            <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB>
            <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
            (m, n, k, dA_array+batch_starting_id, ldda, dB_array+batch_starting_id, lddb, dC_array+batch_starting_id, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}