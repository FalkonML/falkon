from typing import Dict, Optional

import torch
from falkon import FalkonOptions

from falkon.hopt.objectives.exact_objectives import (
    SGPR, NystromGCV, NystromLOOCV, NystromHoldOut, CregNoTrace,
    DeffPenFitTr
)
from falkon.hopt.objectives.stoch_objectives.stoch_new_compreg import StochasticDeffPenFitTr
from falkon.hopt.objectives.stoch_objectives.svgp import SVGP


def init_model(model_type: str,
               data: Dict[str, torch.Tensor],
               penalty_init: torch.Tensor,
               sigma_init: torch.Tensor,
               centers_init: torch.Tensor,
               opt_penalty: bool,
               opt_sigma: bool,
               opt_centers: bool,
               cuda: bool,
               val_pct: Optional[float],
               per_iter_split: Optional[bool],
               cg_tol: Optional[float],
               num_trace_vecs: Optional[int],
               flk_maxiter: Optional[int],
               ):
    flk_opt = FalkonOptions(cg_tolerance=cg_tol, use_cpu=not torch.cuda.is_available(),
                            cg_full_gradient_every=10, cg_epsilon_32=1e-6,
                            cg_differential_convergence=True)

    if model_type == "sgpr":
        model = SGPR(sigma_init=sigma_init, penalty_init=penalty_init, centers_init=centers_init,
                     opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                     cuda=cuda)
    elif model_type == "gcv":
        model = NystromGCV(sigma_init=sigma_init, penalty_init=penalty_init,
                           centers_init=centers_init,
                           opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                           cuda=cuda)
    elif model_type == "loocv":
        model = NystromLOOCV(sigma_init=sigma_init, penalty_init=penalty_init,
                             centers_init=centers_init,
                             opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                             cuda=cuda)
    elif model_type == "holdout":
        if val_pct is None:
            raise ValueError("val_pct must be specified for model_type='holdout'")
        if val_pct <= 0 or val_pct >= 100:
            raise RuntimeError("val_pct must be between 1 and 99")
        model = NystromHoldOut(sigma_init=sigma_init, penalty_init=penalty_init,
                               centers_init=centers_init, opt_centers=opt_centers,
                               opt_sigma=opt_sigma, opt_penalty=opt_penalty, cuda=cuda,
                               val_pct=val_pct, per_iter_split=per_iter_split)
    elif model_type == "creg-notrace":
        model = CregNoTrace(sigma_init=sigma_init, penalty_init=penalty_init,
                            centers_init=centers_init, opt_sigma=opt_sigma,
                            opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda)
    elif model_type == "creg-penfit":
        model = DeffPenFitTr(sigma_init=sigma_init, penalty_init=penalty_init,
                             centers_init=centers_init, opt_sigma=opt_sigma,
                             opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda)
    elif model_type == "stoch-creg-penfit":
        model = StochasticDeffPenFitTr(sigma_init=sigma_init, penalty_init=penalty_init,
                                       centers_init=centers_init, opt_sigma=opt_sigma,
                                       opt_penalty=opt_penalty, opt_centers=opt_centers, cuda=cuda,
                                       flk_opt=flk_opt, num_trace_est=num_trace_vecs,
                                       flk_maxiter=flk_maxiter)
    elif model_type == "svgp":
        model = SVGP(sigma_init=sigma_init, penalty_init=penalty_init, centers_init=centers_init,
                     opt_sigma=opt_sigma, opt_penalty=opt_penalty, opt_centers=opt_centers,
                     cuda=cuda, num_data=data['X'].shape[0], multi_class=data['Y'].shape[1])
    else:
        raise RuntimeError(f"{model_type} model type not recognized!")

    return model
