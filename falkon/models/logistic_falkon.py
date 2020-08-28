import time
from typing import Union, Optional, List

import torch

import falkon
from falkon.gsc_losses import Loss
from falkon.models.model_utils import FalkonBase
from falkon.optim import ConjugateGradient
from falkon.options import *
from falkon.utils import TicToc

__all__ = ("LogisticFalkon", )


class LogisticFalkon(FalkonBase):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty_list: List[float],
                 iter_list: List[int],
                 loss: Loss,
                 M: int,
                 center_selection: Union[str, falkon.center_selection.CenterSelector] = 'uniform',
                 seed: Optional[int] = None,
                 error_fn: Optional[callable] = None,
                 error_every: Optional[int] = 1,
                 options=FalkonOptions(),
                 ):
        super().__init__(kernel, M, center_selection, seed, error_fn, error_every, options)
        self.penalty_list = penalty_list
        self.iter_list = iter_list
        if len(self.iter_list) != len(self.penalty_list):
            raise ValueError("Iteration list must be of same length as penalty list "
                             "(found %d and %d)" % (len(self.iter_list), len(self.penalty_list)))
        self.loss = loss
        self._init_cuda()

    def fit(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            Xts: Optional[torch.Tensor] = None,
            Yts: Optional[torch.Tensor] = None):
        X, Y, Xts, Yts = self._check_fit_inputs(X, Y, Xts, Yts)

        dtype = X.dtype
        self.fit_times_ = []

        t_s = time.time()
        ny_X, ny_Y = self.center_selection.select(X, Y, self.M)
        if self.use_cuda_:
            ny_X = ny_X.pin_memory()

        # beta is the temporary iterative solution
        beta = torch.zeros(ny_X.shape[0], 1, dtype=dtype)
        optim = ConjugateGradient(opt=self.options)
        validation_cback = None
        precond = None
        if self.error_fn is not None and self.error_every is not None:
            def validation_cback(iteration, x, pc, train_time):
                self.fit_times_.append(train_time)
                if iteration % self.error_every != 0:
                    print("Iteration %3d - Elapsed %.1fs" % (iteration, self.fit_times_[-1]), flush=True)
                    return
                err_str = "training" if Xts is None or Yts is None else "validation"
                coeff = pc.invT(x)
                # Compute error: can be train or test;
                if Xts is not None and Yts is not None:
                    pred = self._predict(Xts, ny_X, coeff)
                    err = self.error_fn(Yts, pred)
                    loss = torch.mean(self.loss(Yts, pred)).item()
                else:
                    pred = self._predict(X, ny_X, coeff)
                    err = self.error_fn(Y, pred)
                    loss = torch.mean(self.loss(Y, pred)).item()
                err_name = "error"
                if isinstance(err, tuple) and len(err) == 2:
                    err, err_name = err
                print(f"Iteration {iteration:3d} - Elapsed {self.fit_times_[-1]:.2f}s - "
                      f"{err_str} loss {loss:.4f} - "
                      f"{err_str} {err_name} {err:.4f} ", flush=True)

        t_elapsed = 0.0
        for it, penalty in enumerate(self.penalty_list):
            max_iter = self.iter_list[it]
            print("Iteration %d - penalty %e - sub-iterations %d" % (it, penalty, max_iter), flush=True)

            with TicToc("Preconditioner", self.options.debug):
                if precond is None:
                    precond = falkon.preconditioner.LogisticPreconditioner(
                        self.kernel, self.loss, self.options)
                precond.init(ny_X, ny_Y, beta, penalty, X.shape[0])
            if self.use_cuda_:
                torch.cuda.empty_cache()

            with TicToc("Gradient", self.options.debug):
                # Gradient computation
                knmp_grad, inner_mmv = self.loss.knmp_grad(
                    X, ny_X, Y, precond.invT(beta), opt=self.options)
                grad_p = precond.invAt(precond.invTt(knmp_grad).add_(penalty * beta))

            with TicToc("Optim", self.options.debug):
                # MMV operation for CG
                def mmv(sol):
                    sol_a = precond.invA(sol)
                    knmp_hess = self.loss.knmp_hess(
                        X, ny_X, Y, inner_mmv, precond.invT(sol_a), opt=self.options)
                    return precond.invAt(precond.invTt(knmp_hess).add_(sol_a.mul_(penalty)))
                optim_out = optim.solve(X0=None, B=grad_p, mmv=mmv, max_iter=max_iter, callback=None)
                beta -= precond.invA(optim_out)

            t_elapsed += time.time() - t_s
            if validation_cback is not None:
                validation_cback(it, beta, precond, train_time=t_elapsed)
            t_s = time.time()
        t_elapsed += time.time() - t_s

        if validation_cback is not None:
            validation_cback(len(self.penalty_list), beta, precond, train_time=t_elapsed)
        self.alpha_ = precond.invT(beta)
        self.ny_points_ = ny_X

    def _predict(self, X, ny_points, alpha):
        return self.kernel.mmv(X, ny_points, alpha, opt=self.options)

