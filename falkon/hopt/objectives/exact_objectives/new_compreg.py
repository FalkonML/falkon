from typing import Optional, Dict

import falkon
import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar


class NystromCompReg(HyperoptObjective):
    def __init__(
            self,
            kernel: falkon.kernels.DiffKernel,
            centers_init: torch.Tensor,
            penalty_init: torch.Tensor,
            opt_centers: bool,
            opt_penalty: bool,
            centers_transform: Optional[torch.distributions.Transform] = None,
            pen_transform: Optional[torch.distributions.Transform] = None, ):
        super().__init__(kernel, centers_init, penalty_init,
                         opt_centers, opt_penalty,
                         centers_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, X, Y):
        self.x_train, self.y_train = X.detach(), Y.detach()
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = self.kernel(X, X, diag=True).sum()

        L, A, AAT, LB, c = self._calc_intermediate(X, Y)
        C = torch.linalg.solve_triangular(LB, A, upper=False)  # m * n

        datafit = (torch.square(Y).sum() - torch.square(c / sqrt_var).sum())
        ndeff = (C / sqrt_var).square().sum()
        trace = (Kdiag - torch.trace(AAT))
        trace = trace * datafit / (variance * X.shape[0])
        self._save_losses(ndeff, datafit, trace)

        return ndeff + datafit + trace

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, AAT, LB, c = self._calc_intermediate(self.x_train, self.y_train)
            tmp1 = torch.linalg.solve_triangular(LB.T, c, upper=True)
            tmp2 = torch.linalg.solve_triangular(L.T, tmp1, upper=True)
            kms = self.kernel(self.centers, X)
            return kms.T @ tmp2

    def _calc_intermediate(self, X, Y):
        variance = self.penalty * X.shape[0]

        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)

        L = jittering_cholesky(kmm)
        A = torch.linalg.solve_triangular(L, kmn, upper=False)
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT / variance + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B

        AY = A @ Y  # m*1
        c = torch.linalg.solve_triangular(LB, AY, upper=False)  # m * p

        return L, A, AAT, LB, c

    def _save_losses(self, effective_dimension, data_fit, kernel_trace):
        self.losses = {
            "effective_dimension": effective_dimension.detach(),
            "data_fit": data_fit.detach(),
            "trace": kernel_trace.detach(),
        }

    def __repr__(self):
        return f"NystromCompReg(" \
               f"kernel={self.kernel}, " \
               f"penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})"
