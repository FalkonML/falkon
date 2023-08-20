from typing import Optional, Dict

import torch

import falkon.kernels
from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar


class SGPR(HyperoptObjective):
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

        Kdiag = self.kernel(X, X, diag=True).sum()
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

        # const = 0.5 * X.shape[0] * torch.log(torch.tensor(2 * np.pi, dtype=X.dtype))

        self._save_losses(logdet, datafit, trace)
        return logdet + datafit + trace

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, AAT, LB, c = self._calc_intermediate(self.x_train, self.y_train)
            kms = self.kernel(self.centers, X)
            tmp1 = torch.linalg.solve_triangular(L, kms, upper=False)
            tmp2 = torch.linalg.solve_triangular(LB, tmp1, upper=False)
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

        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)
        L = jittering_cholesky(kmm)

        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.linalg.solve_triangular(L, kmn, upper=False) / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        AY = A @ Y
        c = torch.linalg.solve_triangular(LB, AY, upper=False) / sqrt_var

        return L, A, AAT, LB, c

    def __repr__(self):
        return f"SGPR(" \
               f"kernel={self.kernel}, " \
               f"penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})"
