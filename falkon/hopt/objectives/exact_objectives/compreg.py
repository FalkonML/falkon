import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import NKRRHyperoptObjective
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class CregNoTrace(NKRRHyperoptObjective):
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

        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        self.L = jittering_cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution  # / sqrt_var
        AAT = A @ A.T  # m*n @ n*m = m*m in O(n * m^2), equivalent to kmn @ knm.
        # B = A @ A.T + I
        B = AAT / variance + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = jittering_cholesky(B)  # LB @ LB.T = B
        AY = A @ Y / sqrt_var  # m*1
        self.c = torch.triangular_solve(AY, self.LB, upper=False).solution / sqrt_var  # m*1

        C = torch.triangular_solve(A / sqrt_var, self.LB, upper=False).solution  # m*n

        # Complexity (nystrom-deff)
        datafit = (torch.square(Y).sum() - torch.square(self.c * sqrt_var).sum())
        ndeff = (C.square().sum())

        return ndeff, datafit

    def predict(self, X):
        if self.L is None or self.LB is None or self.c is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            tmp1 = torch.triangular_solve(self.c, self.LB, upper=False, transpose=True).solution
            tmp2 = torch.triangular_solve(tmp1, self.L, upper=False, transpose=True).solution
            kms = full_rbf_kernel(self.centers, X, self.sigma)
            return kms.T @ tmp2

    @property
    def loss_names(self):
        return "nys-deff", "data-fit"

    def __repr__(self):
        return f"CregNoTrace(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
