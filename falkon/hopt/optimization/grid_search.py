import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch

from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.optimization.reporting import pred_reporting, report_losses
from falkon.hopt.utils import get_scalar


# TODO: THIS IS BROKEN (due to attempting to change the parameters of a nn.Module)


@dataclass
class HPGridPoint:
    attributes: Dict[str, Any]
    results: Optional[Dict[str, float]] = None


def set_grid_point(model: HyperoptObjective, grid_point: HPGridPoint):
    for attr_name, attr_val in grid_point.attributes.items():
        setattr(model, attr_name, attr_val)


def run_on_grid(
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    Xts: torch.Tensor,
    Yts: torch.Tensor,
    model: HyperoptObjective,
    grid_spec: List[HPGridPoint],
    minibatch: Optional[int],
    err_fn,
    cuda: bool,
):
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()

    print(f"Starting grid-search on model {model}.")
    print(f"Will run for {len(grid_spec)} points.")

    if minibatch is None or minibatch <= 0:
        minibatch = Xtr.shape[0]
    cum_time = 0
    for i, grid_point in enumerate(grid_spec):
        e_start = time.time()
        set_grid_point(model, grid_point)
        losses = [0.0] * len(model.loss_names)
        for mb_start in range(0, Xtr.shape[0], minibatch):
            Xtr_batch = Xtr[mb_start : mb_start + minibatch, :]
            Ytr_batch = Ytr[mb_start : mb_start + minibatch, :]
            mb_losses = model.hp_loss(Xtr_batch, Ytr_batch)
            for lidx in range(len(mb_losses)):
                losses[lidx] += get_scalar(mb_losses[lidx])
        cum_time += time.time() - e_start
        grid_point.results = pred_reporting(
            model=model,
            Xtr=Xtr,
            Ytr=Ytr,
            Xts=Xts,
            Yts=Yts,
            resolve_model=True,
            err_fn=err_fn,
            epoch=i,
            cum_time=cum_time,
            mb_size=minibatch,
        )
        if not model.losses_are_grads:
            grid_point.results.update(report_losses(losses, model.loss_names, i))
    return grid_spec
