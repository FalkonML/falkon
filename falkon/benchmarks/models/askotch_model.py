import torch
from fast_krr.models import FullKRR
from fast_krr.opts import ASkotchV2

from tqdm import trange


class ASkotchWrapper:
    def __init__(self, Xtr, Ytr, Xts, Yts, block_size, precond_params, kernel_params, lam, task, num_iter, device, log_every=1):
        w0 = torch.zeros((Xtr.shape[0], ), device=device)
        model = FullKRR(
            Xtr, Ytr, Xts, Yts, kernel_params=kernel_params, Ktr_needed=True, lambd=lam, task=task, w0=w0, device=device
        )
        self.opt = ASkotchV2(model=model, block_sz=block_size, precond_params=precond_params)

        self.kern_fn = self.opt.model._get_kernel_fn()
        self.num_iter = num_iter
        self.log_every = log_every

    def fit(self, Xtr, Ytr, Xts, Yts, err_fn):
        for i in trange(1, self.num_iter + 1, desc="Optimization progress"):
            self.opt.step()

    def predict(self, Xtst):
        K_pred = self.kern_fn(Xtst, self.opt.model.x, False) # I m assuming model is FullKRR
        return K_pred @ self.opt.model.w
    
