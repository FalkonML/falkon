from typing import Optional, Dict

import numpy as np
import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective2
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class SGPR(HyperoptObjective2):
    def __init__(
            self,
            centers_init: torch.Tensor,
            sigma_init: torch.Tensor,
            penalty_init: torch.Tensor,
            opt_centers: bool,
            opt_sigma: bool,
            opt_penalty: bool,
            centers_transform: Optional[torch.distributions.Transform] = None,
            sigma_transform: Optional[torch.distributions.Transform] = None,
            pen_transform: Optional[torch.distributions.Transform] = None, ):
        super(SGPR, self).__init__(centers_init, sigma_init, penalty_init,
                                   opt_centers, opt_sigma, opt_penalty,
                                   centers_transform, sigma_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, X, Y):
        self.x_train, self.y_train = X.detach(), Y.detach()

        Kdiag = X.shape[0]
        variance = self.penalty * X.shape[0]
        L, A, AAT, LB, c = self._calc_intermediate(X, Y)

        # Complexity
        logdet = torch.log(torch.diag(LB)).sum()
        logdet += 0.5 * X.shape[0] * torch.log(variance)
        # Data-fit
        datafit = 0.5 * torch.square(Y).sum() / variance
        datafit -= 0.5 * torch.square(c).sum()
        # Traces (minimize)
        trace = 0.5 * Kdiag / variance
        trace -= 0.5 * torch.diag(AAT).sum()

        const = 0.5 * X.shape[0] * torch.log(torch.tensor(2 * np.pi, dtype=X.dtype))

        self._save_losses(logdet, datafit, trace)
        return logdet + datafit + trace

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, AAT, LB, c = self._calc_intermediate(self.x_train, self.y_train)
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            tmp1 = torch.triangular_solve(kms, L, upper=False).solution
            tmp2 = torch.triangular_solve(tmp1, LB, upper=False).solution
            return tmp2.T @ c

    def _save_losses(self, log_det, datafit, trace):
        self.losses = {
            "log_det": log_det.detach(),
            "data_fit": datafit.detach(),
            "trace": trace.detach(),
        }

    def _calc_intermediate(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)

        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        L = jittering_cholesky(kmm)

        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        AY = A @ Y
        c = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var

        return L, A, AAT, LB, c

    def __repr__(self):
        return f"SGPR(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})"
