#include <thread>
#include <stdio.h>

#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;


#define NB 64


/*
  Matrix is size * size (no support for different size than stride).
  Columns are contiguous.
  The size * size grid is subdivided into NB * size blocks (of rows).
  Each block has NB threads, so each thread copies one row into one
  column (transpose).
  Not a particularly efficient implementation!
*/
template <typename scalar_t>
__global__ void copy_simple_kernel_lower(scalar_t *data, int size)
{
	int i = blockIdx.x * NB + threadIdx.x;
	// data iterates across row i, dataT across column i.
	scalar_t *dataT = data;

	if (i < size) {
		data += i;  // Positioned at row i, col 0
		dataT += i * size;  // Positioned at row 0, col i
		scalar_t *row_end = data + i * size;  // row i, col size
		while (data < row_end) {
			*dataT = (scalar_t)*data;
			data += size;
			dataT += 1;
		}
	}
}

// Same as the _lower version, but we copy dataT to data instead!
template <typename scalar_t>
__global__ void copy_simple_kernel_upper(scalar_t *data, int size)
{
	int i = blockIdx.x * NB + threadIdx.x;
	// data iterates across row i, dataT across column i.
	scalar_t *dataT = data;

	if (i < size) {
		data += i;  // Positioned at row i, col 0
		dataT += i * size;  // Positioned at row 0, col i
		scalar_t *row_end = data + i * size;  // row i, col size
		while (data < row_end) {
			*data = (scalar_t)*dataT;
			data += size;
			dataT += 1;
		}
	}
}


template <typename scalar_t>
__global__ void mul_upper_diag(scalar_t *data, int size, scalar_t mul)
{
	int i = blockIdx.x * NB + threadIdx.x;

	if (i < size) {
		data += i * size;
		scalar_t *diag_stop = data + i;
		while (data <= diag_stop) {
			*data *= mul;
			data++;
		}
	}
}


template <typename scalar_t>
__global__ void mul_upper(scalar_t *data, int size, scalar_t mul)
{
	int i = blockIdx.x * NB + threadIdx.x;

	if (i < size) {
		data += i * size;
		scalar_t *diag_stop = data + i;
		while (data < diag_stop) {
			*data *= mul;
			data++;
		}
	}
}


template <typename scalar_t>
__global__ void mul_lower_diag(scalar_t *data, int size, scalar_t mul)
{
	int i = blockIdx.x * NB + threadIdx.x;

	if (i < size) {
		data += i * size + i;
		scalar_t *diag_stop = data + size - i;
		while (data <= diag_stop) {
			*data *= mul;
			data++;
		}
	}
}

template <typename scalar_t>
__global__ void mul_lower(scalar_t *data, int size, scalar_t mul)
{
	int i = blockIdx.x * NB + threadIdx.x;

	if (i < size) {
		data += i * size + i;
		scalar_t *diag_stop = data + size - i;
		while (data < diag_stop) {
			*data *= mul;
			data++;
		}
	}
}


int ceildiv(int dividend, int divisor) {
	int res = dividend / divisor;
	if (dividend % divisor != 0)
		res++;
	return res;
}


torch::Tensor cuda_copy_triang(torch::Tensor &A, bool upper) {
    if (!A.is_cuda()) {
        AT_ERROR("Input A must be a CUDA tensor.");
    }

    bool needs_transpose = false;
    if (A.stride(0) != 1) {
        // Not F-contig (assume C-contig)
        A = torch::transpose(A, 0, 1);
        upper = !upper;
        needs_transpose = true;
    }

    const int nx = A.size(0);
    const int ny = A.size(1);
    const auto scalar_type = A.scalar_type();


    dim3 dimGrid(ceildiv(nx, NB));
    dim3 dimBlock(NB);

    /* Run CUDA kernel */
    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
        scalar_t *data = A.data_ptr<scalar_t>();
	if (upper) {
		copy_simple_kernel_upper<scalar_t><<<dimGrid, dimBlock>>>(data, nx);
	} else {
		copy_simple_kernel_lower<scalar_t><<<dimGrid, dimBlock>>>(data, nx);
	}
    });

    if (needs_transpose) {
	A = torch::transpose(A, 0, 1);
    }
    return A;
}

torch::Tensor cuda_mul_triang(torch::Tensor &A, bool upper, bool preserve_diag, double multiplier) {
    if (!A.is_cuda()) {
        AT_ERROR("Input A must be a CUDA tensor.");
    }
    if (A.stride(0) != 1) {
	upper = !upper;
    }

    const int nx = A.size(0);
    const auto scalar_type = A.scalar_type();
    dim3 dimGrid(ceildiv(nx, NB));
    dim3 dimBlock(NB);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
	scalar_t mul = (scalar_t)multiplier;
	scalar_t *data = A.data_ptr<scalar_t>();
	if (upper && preserve_diag) {
		mul_upper<scalar_t><<<dimGrid, dimBlock>>>(data, nx, mul);
	} else if (upper) {
		mul_upper_diag<scalar_t><<<dimGrid, dimBlock>>>(data, nx, mul);
	} else if (!upper && preserve_diag) {
		mul_lower<scalar_t><<<dimGrid, dimBlock>>>(data, nx, mul);
	} else if (!upper) {
		mul_lower_diag<scalar_t><<<dimGrid, dimBlock>>>(data, nx, mul);
	}
    });
    return A;
}
