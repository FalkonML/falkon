from typing import Optional, Dict

import torch

import falkon.kernels
from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar


class HoldOut(HyperoptObjective):
    def __init__(
            self,
            kernel: falkon.kernels.DiffKernel,
            centers_init: torch.Tensor,
            penalty_init: torch.Tensor,
            opt_centers: bool,
            opt_penalty: bool,
            val_pct: float,
            per_iter_split: bool,
            centers_transform: Optional[torch.distributions.Transform] = None,
            pen_transform: Optional[torch.distributions.Transform] = None,
    ):
        super(HoldOut, self).__init__(kernel, centers_init, penalty_init,
                                      opt_centers, opt_penalty,
                                      centers_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None
        self.per_iter_split = per_iter_split
        self.val_pct = val_pct
        self.tr_indices, self.val_indices = None, None

    def forward(self, X, Y):
        # X_tr, Y_tr are used for predictions. They contain the whole dataset (=retraining)
        self.x_train, self.y_train = X, Y
        if self.tr_indices is None or self.per_iter_split:
            num_val = int(X.shape[0] * self.val_pct)
            all_idx = torch.randperm(X.shape[0])
            self.val_indices = all_idx[:num_val]
            self.tr_indices = all_idx[num_val:]

        Xtr = X[self.tr_indices]
        Xval = X[self.val_indices]
        Ytr = Y[self.tr_indices]
        Yval = Y[self.val_indices]

        kmval = self.kernel(self.centers, Xval)
        alpha = self._calc_intermediate(Xtr, Ytr)
        val_preds = kmval.T @ alpha
        loss = torch.mean(torch.square(Yval - val_preds))

        self._save_losses(loss)
        return loss

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            alpha = self._calc_intermediate(self.x_train, self.y_train)
            kms = self.kernel(self.centers, X)
            return kms.T @ alpha

    @property
    def train_pct(self):
        return 100.0 - self.val_pct

    def _calc_intermediate(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)

        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)
        L = jittering_cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        AYtr = A @ Y
        c = torch.triangular_solve(AYtr, LB, upper=False).solution / sqrt_var

        tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
        alpha = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
        return alpha

    def _save_losses(self, holdout):
        self.losses = {
            "hold-out": holdout.detach(),
        }

    def __repr__(self):
        return f"NystromHoldOut(" \
               f"kernel={self.kernel}, " \
               f"penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]}, " \
               f"val_pct={self.val_pct}, " \
               f"per_iter_split={self.per_iter_split})"
