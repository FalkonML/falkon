#include <thread>
#include <stdio.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>


#define NB 4


/*
  Matrix is size * size (no support for different size than stride).
  Columns are contiguous.
  The size * size grid is subdivided into NB * size blocks (of rows).
  Each block has NB threads, so each thread copies one row into one
  column (transpose).
  Not a particularly efficient implementation!
*/
template <typename scalar_t>
__global__ void copy_simple_kernel_lower(scalar_t *data, const size_t size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		int col_pos = i * size;
		for (int row_pos = i; row_pos < i + i * size; row_pos += size) {
			data[col_pos] = data[row_pos];
			col_pos++;
		}
	}
}

// Same as the _lower version, but we copy dataT to data instead!
template <typename scalar_t>
__global__ void copy_simple_kernel_upper(scalar_t *data, const size_t size)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		int col_pos = i * size;
		for (int row_pos = i; row_pos < i + i * size; row_pos += size) {
			data[row_pos] = data[col_pos];
			col_pos++;
		}
	}
}


template <typename scalar_t>
__global__ void mul_upper_diag(scalar_t *data, const size_t size, const scalar_t mul)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		data += i * size;
		const scalar_t *diag_stop = data + i;
		while (data <= diag_stop) {
			*data *= mul;
			data++;
		}
	}
}


template <typename scalar_t>
__global__ void mul_upper(scalar_t *data, const size_t size, const scalar_t mul)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		data += i * size;
		const scalar_t *diag_stop = data + i;
		while (data < diag_stop) {
			*data *= mul;
			data++;
		}
	}
}


template <typename scalar_t>
__global__ void mul_lower_diag(scalar_t *data, const size_t size, const scalar_t mul)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		data += i * size + i;
		const scalar_t *diag_stop = data + size - i;
		while (data < diag_stop) {
			*data *= mul;
			data++;
		}
	}
}

template <typename scalar_t>
__global__ void mul_lower(scalar_t *data, const size_t size, const scalar_t mul)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
		data += i * size + i;
		const scalar_t *diag_stop = data + size - i;
		data++; // Avoid touching the diagonal
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

    const auto nx = A.size(0);
    const auto ny = A.size(1);
    const auto scalar_type = A.scalar_type();

    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    /* Run CUDA kernel */
    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
	at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
	if (upper) {
		copy_simple_kernel_upper<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx);
	} else {
		copy_simple_kernel_lower<scalar_t><<<dimGrid, dimBlock, 0, stream.stream()>>>(A.data_ptr<scalar_t>(), nx);
	}
    });

    if (needs_transpose) {
	A = torch::transpose(A, 0, 1);
    }
    return A;
}

torch::Tensor cuda_mul_triang(torch::Tensor &A, bool upper, const bool preserve_diag, const double multiplier) {
    if (!A.is_cuda()) {
        AT_ERROR("Input A must be a CUDA tensor.");
    }
    if (A.stride(0) != 1) {
	upper = !upper;
    }

    const auto nx = A.size(0);
    const auto scalar_type = A.scalar_type();
    const dim3 dimGrid(ceildiv(nx, NB));
    const dim3 dimBlock(NB);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "dispatch", [&] {
	const scalar_t mul = (scalar_t)multiplier;
	if (upper && preserve_diag) {  // U, preserve
		mul_upper<scalar_t><<<dimGrid, dimBlock>>>(A.data_ptr<scalar_t>(), nx, mul);
	} else if (upper) {            // U, no-preserve
		mul_upper_diag<scalar_t><<<dimGrid, dimBlock>>>(A.data_ptr<scalar_t>(), nx, mul);
	} else if (preserve_diag) {    // L, preserve
		mul_lower<scalar_t><<<dimGrid, dimBlock>>>(A.data_ptr<scalar_t>(), nx, mul);
	} else {                       // L, no-preserve
		mul_lower_diag<scalar_t><<<dimGrid, dimBlock>>>(A.data_ptr<scalar_t>(), nx, mul);
	}
    });
    return A;
}
