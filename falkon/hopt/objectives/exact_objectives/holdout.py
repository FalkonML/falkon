import torch

from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import NKRRHyperoptObjective
from falkon.hopt.utils import full_rbf_kernel, get_scalar


class NystromHoldOut(NKRRHyperoptObjective):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            val_pct: float,
            per_iter_split: bool,
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

        self.per_iter_split = per_iter_split
        self.val_pct = val_pct
        self.tr_indices, self.val_indices = None, None

        self.alpha = None

    def hp_loss(self, X, Y):
        if self.tr_indices is None or self.per_iter_split:
            num_val = int(X.shape[0] * self.val_pct)
            all_idx = torch.randperm(X.shape[0])
            self.val_indices = all_idx[:num_val]
            self.tr_indices = all_idx[num_val:]

        Xtr = X[self.tr_indices]
        Xval = X[self.val_indices]
        Ytr = Y[self.tr_indices]
        Yval = Y[self.val_indices]

        variance = self.penalty * Xtr.shape[0]
        sqrt_var = torch.sqrt(variance)

        kmn = full_rbf_kernel(self.centers, Xtr, self.sigma)
        kmm = full_rbf_kernel(self.centers, self.centers, self.sigma)
        kmval = full_rbf_kernel(self.centers, Xval, self.sigma)

        L = jittering_cholesky(kmm)   # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=Xtr.device, dtype=Xtr.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        AYtr = A @ Ytr
        c = torch.triangular_solve(AYtr, LB, upper=False).solution / sqrt_var

        tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
        self.alpha = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
        val_preds = kmval.T @ self.alpha

        return (torch.mean(torch.square(Yval - val_preds)), )

    def predict(self, X):
        if self.alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")
        kms = full_rbf_kernel(self.centers, X, self.sigma)
        return kms.T @ self.alpha

    @property
    def loss_names(self):
        return ("val-mse", )

    @property
    def train_pct(self):
        return 100.0 - self.val_pct

    def __repr__(self):
        return f"NystromHoldOut(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, opt_centers={self.opt_centers}, " \
               f"opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}, val_pct={self.val_pct}, " \
               f"per_iter_split={self.per_iter_split})"
