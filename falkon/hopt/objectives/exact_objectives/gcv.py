import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import NKRRHyperoptObjective
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class NystromGCV(NKRRHyperoptObjective):
    r"""
    GCV objective is

    ..math:

        \dfrac{\dfrac{1}{n} \lVert (I - \widetilde{K}_\lambda \widetilde{K}) Y \rVert^2}
              {\Big(\frac{1}{n} \mathrm{Tr}(I - \widetilde{K}_\lambda \widetilde{K}) \Big)}

    We must compute the two terms denoted as the numerator and the denominator.
    Using the usual names for matrix variable substitutions (taken from gpflow code), we have that
    the numerator can be computed as

    ..math:

        \dfrac{1}{n} \lVert (I - A^\top \mathrm{LB}^{-\top} \mathrm{LB}^{-1} A) Y \rVert^2

    We compute the terms inside the norm first, from right to left using matrix-vector multiplications
    and triangular solves. Finally we compute the norm.

    The denominator is far less efficient to compute, since it requires working with m*n matrices.
    It can be expressed in terms of the same matrices as above:

    ..math:

        \Big( \frac{1}{n} (\mathrm{Tr}(I} - \lVert \mathrm{LB}^{-1}A \rVert_F^2 \Big)^2

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

        self.L, self.LB, self.d = None, None, None

    def hp_loss(self, X, Y):
        # Like with LOOCV we are virtually using an estimator trained with n - 1 points.
        variance = self.penalty * (X.shape[0] - 1)
        sqrt_var = torch.sqrt(variance)

        kmn = full_rbf_kernel(self.centers, X, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        self.L = jittering_cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, self.L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        self.LB = jittering_cholesky(B)  # LB @ LB.T = B

        AY = A @ Y
        # numerator is (1/n)*||(I - A.T @ LB^{-T} @ LB^{-1} @ A) @ Y||^2
        # compute A.T @ LB^{-T} @ LB^{-1} @ A @ Y
        tmp1 = torch.triangular_solve(AY, self.LB, upper=False).solution
        tmp2 = torch.triangular_solve(tmp1, self.LB, upper=False, transpose=True).solution
        self.d = tmp2 / sqrt_var  # only for predictions
        tmp3 = Y - A.T @ tmp2
        numerator = torch.square(tmp3)

        # Denominator (1/n * Tr(I - H))^2
        C = torch.triangular_solve(A, self.LB, upper=False).solution
        denominator = (1 - torch.square(C).sum() / X.shape[0]) ** 2

        return ((numerator / denominator).mean(), )

    def predict(self, X):
        if self.L is None or self.LB is None or self.d is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        tmp1 = torch.triangular_solve(self.d, self.L, upper=False, transpose=True).solution
        return kms.T @ tmp1

    @property
    def loss_names(self):
        return ("gcv",)

    def __repr__(self):
        return f"NystromGCV(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty})"
