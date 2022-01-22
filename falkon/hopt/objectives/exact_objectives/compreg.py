from typing import Optional, Dict

import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective2
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class CompReg(HyperoptObjective2):
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
        super(CompReg, self).__init__(centers_init, sigma_init, penalty_init,
                                      opt_centers, opt_sigma, opt_penalty,
                                      centers_transform, sigma_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def _calc_intermediate(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)

        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        L = jittering_cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT / variance + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        AY = A @ Y / sqrt_var  # m*1
        c = torch.triangular_solve(AY, LB, upper=False).solution / sqrt_var  # m*1

        return L, A, LB, c

    def forward(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)

        L, A, LB, c = self._calc_intermediate(X, Y)
        C = torch.triangular_solve(A / sqrt_var, LB, upper=False).solution  # m*n

        ndeff = (C.square().sum())
        datafit = (torch.square(Y).sum() - torch.square(c * sqrt_var).sum())
        self._save_losses(ndeff, datafit)

        return ndeff, datafit

    def _save_losses(self, effective_dimension, data_fit):
        self.losses = {
            "effective_dimension": effective_dimension.detach(),
            "data_fit": data_fit.detach(),
        }

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, LB, c = self._calc_intermediate(self.x_train, self.y_train)
            tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
            tmp2 = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ tmp2

    def __repr__(self):
        return f"CregNoTrace(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})"
