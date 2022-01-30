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
                       backward: torch.Tensor,
                       retain_graph: bool,
                       allow_unused: bool) -> Tuple[Optional[torch.Tensor], ...]:
    assert len(inputs) <= len(inputs_need_grad)
    needs_grad = []
    for i in range(len(inputs)):
        if inputs_need_grad[i]:
            needs_grad.append(inputs[i])
    grads = torch.autograd.grad(
        backward, needs_grad, retain_graph=retain_graph, allow_unused=allow_unused)
    j = 0
    results = []
    for i in range(len(inputs_need_grad)):
        if inputs_need_grad[i]:
            results.append(grads[j])
            j += 1
        else:
            results.append(None)
    return tuple(results)


def calc_grads(ctx, backward, num_diff_args):
    return calc_grads_tensors(ctx.saved_tensors, ctx.needs_input_grad, backward,
                              retain_graph=True, allow_unused=True)
