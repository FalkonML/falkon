import time
from typing import Union, Optional, List

import torch
from sklearn import base

import falkon
from falkon.gsc_losses import Loss
from falkon.optim import ConjugateGradient
from falkon.options import *
from falkon.utils import decide_cuda, devices, TicToc
from falkon.utils.helpers import check_same_dtype
from .falkon import check_random_generator

__all__ = ("LogisticFalkon", )


class LogisticFalkon(base.BaseEstimator):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty_list: List[float],
                 iter_list: List[int],
                 loss: Loss,
                 M: int,
                 center_selection: Union[str, falkon.center_selection.NySel] = 'uniform',
                 seed: Optional[int] = None,
                 error_fn: Optional[callable] = None,
                 error_every: Optional[int] = 1,

                 options=FalkonOptions(),
                 ):
        self.kernel = kernel
        self.penalty_list = penalty_list
        self.iter_list = iter_list
        if len(self.iter_list) != len(self.penalty_list):
            raise ValueError("Iteration list must be of same length as penalty list "
                             "(found %d and %d)" % (len(self.iter_list), len(self.penalty_list)))
        self.M = M
        self.seed = seed
        self.loss = loss
        if self.seed is not None:
            torch.manual_seed(self.seed)  # Works for both CPU and GPU
        self.random_state_ = check_random_generator(self.seed)

        self.error_fn = error_fn
        self.error_every = error_every
        # Options
        self.options = options
        self._cg_options = options.get_conjgrad_options()
        self._keops_options = options.get_keops_options()
        self._pc_options = options.get_pc_options()
        self._cholesky_opt = options.get_chol_options()
        self._lauum_opt = options.get_lauum_options()
        self._base_opt = options.get_base_options()

        self.use_cuda_ = decide_cuda(self.options)
        self.alpha_ = None
        self.ny_points_ = None
        self.fit_times_ = None

        if isinstance(center_selection, str):
            if center_selection.lower() == 'uniform':
                self.center_selection = falkon.center_selection.UniformSel(
                    self.random_state_)
            else:
                raise ValueError(f'Center selection "{center_selection}" is not valid.')
        else:
            self.center_selection = center_selection

        self._init_cuda()

    def _init_cuda(self):
        if self.use_cuda_:
            torch.cuda.init()
            from falkon.cuda import initialization
            initialization.init(self._base_opt)
            self.num_gpus = devices.num_gpus(self.options)

    def fit(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            Xts: Optional[torch.Tensor] = None,
            Yts: Optional[torch.Tensor] = None):
        if X.size(0) != Y.size(0):
            raise ValueError("X and Y must have the same number of "
                             "samples (found %d and %d)" %
                             (X.size(0), Y.size(0)))
        if Y.dim() == 1:
            Y = torch.unsqueeze(Y, 1)
        if Y.dim() != 2:
            raise ValueError("Y is expected 1D or 2D. Found %dD." % (Y.dim()))
        if not check_same_dtype(X, Y):
            raise TypeError("X and Y must have the same data-type.")

        dtype = X.dtype
        self.fit_times_ = []

        t_s = time.time()
        ny_X, ny_Y = self.center_selection.select(X, Y, self.M)
        if self.use_cuda_:
            ny_X = ny_X.pin_memory()

        # beta is the temporary iterative solution
        beta = torch.zeros(ny_X.shape[0], 1, dtype=dtype)
        optim = ConjugateGradient(opt=self.options)
        cback = None
        precond = None
        if self.error_fn is not None and self.error_every is not None:
            def cback(it, x, pc, train_time):
                self.fit_times_.append(train_time)
                if it % self.error_every != 0:
                    print("Iteration %3d - Elapsed %.1fs" % (it, self.fit_times_[-1]), flush=True)
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
                print(f"Iteration {it:3d} - Elapsed {self.fit_times_[-1]:.2f}s - "
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

            # Callback
            def mmv(sol):
                sol_a = precond.invA(sol)
                knmp_hess = self.loss.knmp_hess(
                    X, ny_X, Y, inner_mmv, precond.invT(sol_a), opt=self.options)
                return precond.invAt(precond.invTt(knmp_hess).add_(sol_a.mul_(penalty)))

            with TicToc("Optim", self.options.debug):
                optim_out = optim.solve(X0=None, B=grad_p, mmv=mmv, max_iter=max_iter, callback=None)
                beta -= precond.invA(optim_out)

            t_elapsed += time.time() - t_s
            cback(it, beta, precond, train_time=t_elapsed)
            t_s = time.time()
        t_elapsed += time.time() - t_s

        cback(len(self.penalty_list), beta, precond, train_time=t_elapsed)
        self.alpha_ = precond.invT(beta)
        self.ny_points_ = ny_X

    def _predict(self, X, ny_points, alpha):
        return self.kernel.mmv(X, ny_points, alpha, opt=self.options)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the outputs on given test points.

        Parameters
        -----------
        X : torch.Tensor (2D)
            Tensor of test data points, of shape [num_samples, num_dimensions].
            If X is in Fortran order (i.e. column-contiguous) then we can avoid
            an extra copy of the data.

        Returns
        --------
        predictions : torch.Tensor (2D)
            Prediction tensor of shape [num_samples, num_outputs] for all
            data points.
        """
        if self.alpha_ is None or self.ny_points_ is None:
            raise RuntimeError(
                "Falkon has not been trained. `predict` must be called after `fit`.")

        return self._predict(X, self.ny_points_, self.alpha_)

    def __repr__(self, **kwargs):
        return super().__repr__(N_CHAR_MAX=5000)
