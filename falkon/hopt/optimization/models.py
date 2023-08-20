"""
This file contains a helper function to initialize one of several
hyperparameter optimization objectives.
"""
from typing import Dict, Optional

import torch

import falkon.kernels
from falkon import FalkonOptions

from falkon.hopt.objectives import (
    SGPR,
    GCV,
    LOOCV,
    HoldOut,
    CompReg,
    NystromCompReg,
    StochasticNystromCompReg,
)


def init_model(
    model_type: str,
    data: Dict[str, torch.Tensor],
    kernel: falkon.kernels.DiffKernel,
    penalty_init: torch.Tensor,
    centers_init: torch.Tensor,
    opt_penalty: bool,
    opt_centers: bool,
    cuda: bool,
    val_pct: Optional[float],
    per_iter_split: Optional[bool],
    cg_tol: Optional[float],
    num_trace_vecs: Optional[int],
    flk_maxiter: Optional[int],
):
    flk_opt = FalkonOptions(
        cg_tolerance=cg_tol,
        use_cpu=not torch.cuda.is_available(),
        cg_full_gradient_every=10,
        cg_epsilon_32=1e-6,
        cg_differential_convergence=True,
    )

    if model_type == "sgpr":
        model = SGPR(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_penalty=opt_penalty,
            opt_centers=opt_centers,
        )
    elif model_type == "gcv":
        model = GCV(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_penalty=opt_penalty,
            opt_centers=opt_centers,
        )
    elif model_type == "loocv":
        model = LOOCV(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_penalty=opt_penalty,
            opt_centers=opt_centers,
        )
    elif model_type == "holdout":
        if val_pct is None:
            raise ValueError("val_pct must be specified for model_type='holdout'")
        if val_pct <= 0 or val_pct >= 100:
            raise RuntimeError("val_pct must be between 1 and 99")
        model = HoldOut(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_centers=opt_centers,
            opt_penalty=opt_penalty,
            val_pct=val_pct,
            per_iter_split=per_iter_split,
        )
    elif model_type == "creg-notrace":
        model = CompReg(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_penalty=opt_penalty,
            opt_centers=opt_centers,
        )
    elif model_type == "creg-penfit":
        model = NystromCompReg(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_penalty=opt_penalty,
            opt_centers=opt_centers,
        )
    elif model_type == "stoch-creg-penfit":
        model = StochasticNystromCompReg(
            kernel=kernel,
            penalty_init=penalty_init,
            centers_init=centers_init,
            opt_penalty=opt_penalty,
            opt_centers=opt_centers,
            flk_opt=flk_opt,
            num_trace_est=num_trace_vecs,
            flk_maxiter=flk_maxiter,
        )
    else:
        raise RuntimeError(f"{model_type} model type not recognized!")

    if cuda:
        model = model.cuda()

    return model
