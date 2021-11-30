import numpy as np
import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import NKRRHyperoptObjective
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class SGPR(NKRRHyperoptObjective):
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
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)
        Kdiag = X.shape[0]

        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        self.L = jittering_cholesky(kmm)

        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = jittering_cholesky(B)  # LB @ LB.T = B
        AY = A @ Y
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution / sqrt_var

        # Complexity
        logdet = torch.log(torch.diag(self.LB)).sum()
        logdet += 0.5 * X.shape[0] * torch.log(variance)
        # Data-fit
        datafit = 0.5 * torch.square(Y).sum() / variance
        datafit -= 0.5 * torch.square(self.c).sum()
        # Traces (minimize)
        trace = 0.5 * Kdiag / variance
        trace -= 0.5 * torch.diag(AAT).sum()

        const = 0.5 * X.shape[0] * torch.log(torch.tensor(2 * np.pi, dtype=X.dtype))

        return logdet, datafit, trace

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(kms, self.L, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False).solution
        return tmp2.T @ self.c

    @property
    def loss_names(self):
        return "log-det", "data-fit", "trace"

    def __repr__(self):
        return f"SGPR(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
