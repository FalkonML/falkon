from typing import Optional

import torch

from falkon import c_ext


class SquareNormFn(torch.autograd.Function):
    @staticmethod
    def forward(x, dim: int, keepdim: bool):
        with torch.no_grad():
            out = c_ext.square_norm(x, dim, keepdim)
        return out

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, dim, keepdim = inputs

        ctx.save_for_backward(x)
        ctx.x = x  # saved for jvp
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = (x * 2) * grad_output
        return grad_input, None, None

    @staticmethod
    def jvp(ctx, grad_x, dim, keepdim):
        x = ctx.x
        del ctx.x
        return (2 * x * grad_x).sum(ctx.dim, keepdim=ctx.keepdim)

    @staticmethod
    def vmap(info, in_dims, *fwd_args):
        x, dim, keepdim = fwd_args
        if in_dims[0] is None:  # no vmap
            return SquareNormFn.forward(x, dim, keepdim), None
        if in_dims[0] <= dim:
            return SquareNormFn.forward(x, dim + 1, keepdim), in_dims[0]
        else:
            return SquareNormFn.forward(x, dim, keepdim), in_dims[0]


def square_norm(mat: torch.Tensor, dim: int, keepdim: Optional[bool] = None) -> torch.Tensor:
    return SquareNormFn.apply(mat, dim, keepdim)


if __name__ == "__main__":
    mat = torch.randn(5, 2).double().requires_grad_()
    torch.autograd.gradcheck(
        lambda m: square_norm(m, 0, False),
        (mat,),
        check_undefined_grad=True,
        check_grad_dtypes=True,
        check_batched_grad=True,
        check_batched_forward_grad=True,
        check_forward_ad=True,
        check_backward_ad=True,
    )
