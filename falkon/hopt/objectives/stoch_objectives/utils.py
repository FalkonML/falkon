from typing import Sequence, Tuple, Optional

import torch


def init_random_vecs(n, t, dtype, device, gaussian_random: bool):
    if gaussian_random:
        Z = torch.randn(n, t, dtype=dtype, device=device)
    else:
        Z = torch.empty(n, t, dtype=dtype, device=device).bernoulli_().mul_(2).sub_(1)
    return Z


def calc_grads_tensors(inputs: Sequence[torch.Tensor],
                       inputs_need_grad: Sequence[bool],
                       num_nondiff_inputs: int,
                       output: torch.Tensor,
                       retain_graph: bool,
                       allow_unused: bool) -> Tuple[Optional[torch.Tensor], ...]:
    """

    Parameters
    ----------
    inputs
        Sequence of tensors with respect to which the gradient needs computing
    inputs_need_grad
        Sequence of booleans, stating whether the inputs need the gradient computation.
        This sequence corresponds to ctx.needs_input_grad hence it includes all inputs
        to some nn.Function, not just the differentiable inputs (which are passed in the `inputs`
        parameter).
        Hence `len(inputs_need_grad) != len(inputs)`. To make the code work, the inputs to the
        nn.Function we are dealing with must be organized such that the non-differentiable inputs
        come before the potentially differentiable inputs!
    num_nondiff_inputs: int
        The number of non-differentiable inputs to the nn.Function.
    output
        output of the differentiated function
    retain_graph
        See corresponding option in `torch.autograd.grad`
    allow_unused
        See corresponding option in `torch.autograd.grad`

    Returns
    -------
    The gradients of `output` with respect to the sequence of inputs. If an input does not require
    gradient, the corresponding gradient in the result will be set to `None`.
    """
    assert len(inputs) <= len(inputs_need_grad)

    saved_idx = 0
    needs_grad = []
    for i, i_grad in enumerate(inputs_need_grad):
        if i_grad:
            needs_grad.append(inputs[saved_idx])
        if i >= num_nondiff_inputs:
            saved_idx += 1

    grads = torch.autograd.grad(
        output, needs_grad, retain_graph=retain_graph, allow_unused=allow_unused)

    grads_idx = 0
    results = []
    for i, i_grad in enumerate(inputs_need_grad):
        if i_grad:
            results.append(grads[grads_idx])
            grads_idx += 1
        else:
            results.append(None)
    return tuple(results)


def calc_grads(ctx, output, num_nondiff_inputs):
    return calc_grads_tensors(ctx.saved_tensors, ctx.needs_input_grad, num_nondiff_inputs,
                              output, retain_graph=True, allow_unused=True)
