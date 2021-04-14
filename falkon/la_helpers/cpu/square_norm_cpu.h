#pragma once

#include <torch/extension.h>

template <typename scalar_t>
void square_vector_norm_cpu_impl(at::TensorIterator& iter);

