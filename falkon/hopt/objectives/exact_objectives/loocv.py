from typing import Optional, Dict

import torch

import falkon.kernels
from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar


class LOOCV(HyperoptObjective):
    r"""
    LOOCV objective is to minimize the PRESS error:

    .. math::

        \min \sum_{i=1}^n \big(e_{(i)}\big)_i^2

    whose components are calculated as

    .. math::

        \big(e_{(i)}\big)_i = (y_i - f(\alpha_{(i)})_i) = \dfrac{y_i - f(\alpha)_i}{1 - S_{ii}}

    where

    .. math::

        S = K_{nm} (\lambda K_{mm} + K_{nm}^\top K_{nm})^{-1} K_{nm}^\top

    So we first need to compute the N-KRR solution :math:`\alpha` and then the diagonal of matrix
    :math:`S`. For simplicity we use a direct solver the N-KRR problem, although Falkon could easily
    be substituted in.

    """

    def __init__(
            self,
            kernel: falkon.kernels.DiffKernel,
            centers_init: torch.Tensor,
            penalty_init: torch.Tensor,
            opt_centers: bool,
            opt_penalty: bool,
            centers_transform: Optional[torch.distributions.Transform] = None,
            pen_transform: Optional[torch.distributions.Transform] = None, ):
        super(LOOCV, self).__init__(kernel, centers_init, penalty_init,
                                    opt_centers, opt_penalty,
                                    centers_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, X, Y):
        self.x_train, self.y_train = X.detach(), Y.detach()

        variance = self.penalty * (X.shape[0] - 1)
        sqrt_var = torch.sqrt(variance)
        L, A, LB, C, c = self._calc_intermediate(X, Y)
        diag_s = (C.square()).sum(0)  # == torch.diag(C.T @ C)  n, 1
        # Now f(\alpha) = C.T @ C @ Y
        d = C.T @ c * sqrt_var

        num = Y - d
        den = 1.0 - diag_s
        loss = (num / den).square().mean(0).mean()
        self._save_losses(loss)
        return loss

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, LB, C, c = self._calc_intermediate(self.x_train, self.y_train)
            # Predictions are handled directly.
            tmp1 = torch.linalg.solve_triangular(LB.T, c, upper=True)
            tmp2 = torch.linalg.solve_triangular(L.T, tmp1, upper=True)
            kms = self.kernel(self.centers, X)
            return kms.T @ tmp2

    def _calc_intermediate(self, X, Y):
        # - 1 due to LOOCV effect.
        variance = self.penalty * (X.shape[0] - 1)
        sqrt_var = torch.sqrt(variance)

        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)
        L = jittering_cholesky(kmm)  # L @ L.T = kmm
        A = torch.linalg.solve_triangular(L, kmn, upper=False) / sqrt_var  # m, n
        AAT = A @ A.T  # m, m
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)  # m, m
        LB = jittering_cholesky(B)  # LB @ LB.T = B  m, m

        # C = LB^{-1} A
        C = torch.linalg.solve_triangular(LB, A, upper=False)  # m, n

        c = C @ Y / sqrt_var
        return L, A, LB, C, c

    def _save_losses(self, loocv_loss):
        self.losses = {
            "loocv": loocv_loss.detach(),
        }

    def __repr__(self):
        return f"NystromLOOCV(" \
               f"kernel={self.kernel}, " \
               f"penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})"
