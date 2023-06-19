#include "square_norm.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace falkon {
namespace ops {

at::Tensor square_norm(
    const at::Tensor &self,
    int64_t dim,
    bool keepdim) {

    static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("falkon::square_norm", "")
                       .typed<decltype(square_norm)>();
  return op.call(
      self, dim, keepdim
  );
}

TORCH_LIBRARY_FRAGMENT(falkon, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "falkon::square_norm(Tensor self, int dim, bool keepdim=False) -> Tensor"));
}

} // namespace ops
} // namespace falkon
