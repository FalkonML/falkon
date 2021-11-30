import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import NKRRHyperoptObjective
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class NystromLOOCV(NKRRHyperoptObjective):
    r"""
    LOOCV objective is to minimize the PRESS error:

    ..math:

        \min \sum_{i=1}^n \big(e_{(i)}\big)_i^2

    whose components are calculated as

    ..math:

        \big(e_{(i)}\big)_i = (y_i - f(\alpha_{(i)})_i) = \dfrac{y_i - f(\alpha)_i}{1 - S_{ii}}

    where

    ..math:

        S = K_{nm} (\lambda K_{mm} + K_{nm}^\top K_{nm})^{-1} K_{nm}^\top

    So we first need to compute the N-KRR solution :math:`\alpha` and then the diagonal of matrix
    :math:`S`. For simplicity we use a direct solver the N-KRR problem, although Falkon could easily
    be substituted in.

    """
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
    ):
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        if opt_sigma:
            self.register_parameter("sigma", self.sigma_.requires_grad_(True))
        if opt_penalty:
            self.register_parameter("penalty", self.penalty_.requires_grad_(True))
        if opt_centers:
            self.register_parameter("centers", self.centers_.requires_grad_(True))

        self.L, self.LB, self.c = None, None, None

    def hp_loss(self, X, Y):
        # - 1 due to LOOCV effect.
        variance = self.penalty * (X.shape[0] - 1)
        sqrt_var = torch.sqrt(variance)

        m = self.centers.shape[0]
        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        self.L = jittering_cholesky(kmm)   # L @ L.T = kmm
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var  # m, n
        AAT = A @ A.T   # m, m
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)  # m, m
        self.LB = jittering_cholesky(B)  # LB @ LB.T = B  m, m

        # C = LB^{-1} A
        C = torch.triangular_solve(A, self.LB, upper=False).solution  # m, n
        diag_s = (C.square()).sum(0)  # == torch.diag(C.T @ C)  n, 1

        self.c = C @ Y / sqrt_var
        # Now f(\alpha) = C.T @ C @ Y
        d = C.T @ self.c * sqrt_var

        num = Y - d
        den = 1.0 - diag_s
        return ((num / den).square().mean(0).mean(), )

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        # Predictions are handled directly.
        tmp1 = torch.triangular_solve(self.c, self.LB, upper=False, transpose=True).solution
        tmp2 = torch.triangular_solve(tmp1, self.L, upper=False, transpose=True).solution
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        return kms.T @ tmp2

    @property
    def loss_names(self):
        return ("loocv", )

    def __repr__(self):
        return f"NystromLOOCV(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
