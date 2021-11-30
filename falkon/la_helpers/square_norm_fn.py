import torch
from falkon.c_ext import square_norm

__all__ = (
    "SquareNormFunction",
    "square_norm_diff",
)

# TODO: This function should ideally be implemented as an autograd.Function in Cpp.


# noinspection PyMethodOverriding
class SquareNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat: torch.Tensor, dim: int, keepdim: bool = False):
        if mat.requires_grad:
            ctx.save_for_backward(mat)
        ctx.dim, ctx.keepdim = dim, keepdim
        return square_norm(mat, dim, keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        mat_grad = None
        if ctx.needs_input_grad[0]:
            mat = ctx.saved_tensors[0]
            if not ctx.keepdim:
                grad_output = grad_output.unsqueeze(ctx.dim)
            mat_grad = mat.mul(2).mul_(grad_output)

        return (mat_grad, None, None)


def square_norm_diff(mat: torch.Tensor, dim: int, keepdim: bool = False):
    return SquareNormFunction.apply(mat, dim, keepdim)
