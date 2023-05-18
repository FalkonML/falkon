from typing import Optional, Dict

import torch

import falkon.kernels
from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar


class GCV(HyperoptObjective):
    r"""
    GCV objective is

    .. math::

        \dfrac{\dfrac{1}{n} \lVert (I - \widetilde{K}_\lambda \widetilde{K}) Y \rVert^2}
              {\frac{1}{n} \mathrm{Tr}(I - \widetilde{K}_\lambda \widetilde{K}) }

    We must compute the two terms denoted as the numerator and the denominator.
    Using the usual names for matrix variable substitutions (taken from gpflow code), we have that
    the numerator can be computed as

    .. math::

        \dfrac{1}{n} \lVert (I - A^\top \mathrm{LB}^{-\top} \mathrm{LB}^{-1} A) Y \rVert^2

    We compute the terms inside the norm first, from right to left using matrix-vector multiplications
    and triangular solves. Finally we compute the norm.

    The denominator is far less efficient to compute, since it requires working with m*n matrices.
    It can be expressed in terms of the same matrices as above:

    .. math::

        \Big( \frac{1}{n} (\mathrm{Tr}(I) - \lVert \mathrm{LB}^{-1}A \rVert_F^2 ) \Big)^2

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
        super(GCV, self).__init__(kernel, centers_init, penalty_init,
                                  opt_centers, opt_penalty,
                                  centers_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def _calc_intermediate(self, X, Y):
        # Like with LOOCV we are virtually using an estimator trained with n - 1 points.
        variance = self.penalty * (X.shape[0] - 1)
        sqrt_var = torch.sqrt(variance)

        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)
        L = jittering_cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.linalg.solve_triangular(L, kmn, upper=False) / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B

        AY = A @ Y
        # numerator is (1/n)*||(I - A.T @ LB^{-T} @ LB^{-1} @ A) @ Y||^2
        # compute A.T @ LB^{-T} @ LB^{-1} @ A @ Y
        tmp1 = torch.linalg.solve_triangular(LB, AY, upper=False)
        tmp2 = torch.linalg.solve_triangular(LB.T, tmp1, upper=True)
        d = tmp2
        return L, A, LB, d

    def forward(self, X, Y):
        self.x_train, self.y_train = X.detach(), Y.detach()

        L, A, LB, d = self._calc_intermediate(X, Y)
        tmp3 = Y - A.T @ d
        numerator = torch.square(tmp3)
        # Denominator (1/n * Tr(I - H))^2
        C = torch.linalg.solve_triangular(LB, A, upper=False)
        denominator = (1 - torch.square(C).sum() / X.shape[0]) ** 2
        loss = (numerator / denominator).mean()

        self._save_losses(loss)
        return loss

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, LB, d = self._calc_intermediate(self.x_train, self.y_train)
            sqrt_var = torch.sqrt(self.penalty * (X.shape[0] - 1))
            kms = self.kernel(self.centers, X)
            tmp1 = torch.linalg.solve_triangular(L.T, d / sqrt_var, upper=True)
            return kms.T @ tmp1

    def _save_losses(self, gcv):
        self.losses = {
            "GCV": gcv.detach(),
        }

    def __repr__(self):
        return f"GCV(" \
               f"kernel={self.kernel}, " \
               f"penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})"
