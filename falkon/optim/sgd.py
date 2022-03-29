import functools
import time
from contextlib import ExitStack
from typing import Optional, Callable

import torch
from torch.utils import data as tc_data

import falkon
from falkon.mmv_ops.fmmv_incore import incore_fdmmv
from falkon.optim import Optimizer, StopOptimizationException
from falkon.options import GDOptions, FalkonOptions
from falkon.utils import TicToc


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate: float, momentum: Optional[float] = None):
        super().__init__()
        self.num_iter = None
        self.learning_rate = learning_rate
        if momentum < 0 or momentum > 1:
            raise ValueError(f"Momentum must be between 0 and 1. Given {momentum}")
        if momentum == 0:
            momentum = None
        self.momentum = momentum
        self.last_upd = None

    def solve(self,
              X0: torch.Tensor,
              data: tc_data.DataLoader,
              gradient_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
              max_iter: int,
              params: GDOptions,
              callback: Optional[Callable[[int, torch.Tensor, float], None]] = None) -> torch.Tensor:
        t_start = time.time()

        tol = params.gd_tolerance ** 2
        e_train = time.time() - t_start

        last_upd = self.last_upd
        lr = self.learning_rate

        dl_iter = iter(data)
        for self.num_iter in range(max_iter):
            with TicToc("SGD-Iter", debug=False):
                t_start = time.time()
                data_x, data_y = next(dl_iter)
                data_x = data_x.to(device=X0.device)
                data_y = data_y.to(device=X0.device)
                # Momentum update
                update = (-lr) * gradient_step(X0, data_x, data_y)
                if last_upd is not None:
                    update.add_(last_upd * self.momentum)
                X = X0 + update
                if self.momentum is not None:
                    last_upd = update

                diff = (X - X0).square().sum(dim=0).mean()
                if diff < tol:
                    break
                X0 = X
            e_iter = time.time() - t_start
            e_train += e_iter
            with TicToc("SGD-callback", debug=False):
                if callback is not None:
                    try:
                        callback(self.num_iter + 1, X, e_train)
                    except StopOptimizationException as e:
                        print(f"Optimization stopped from callback: {e.message}")
                        break
        self.last_upd = last_upd
        return X0


class FalkonSGD(Optimizer):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 learning_rate: float,
                 momentum: Optional[float],
                 weight_fn=None):
        super().__init__()
        if weight_fn is not None:
            raise NotImplementedError("Gradient solver does not support weight function. "
                                      "Use the CG solver instead.")
        self.kernel = kernel
        self.optimizer = StochasticGradientDescent(learning_rate=learning_rate, momentum=momentum)

    def sgd_iter(self, sol, X, Y, M, penalty, prec, n, opt):
        b = X.shape[0]
        with TicToc("SGD-update-step", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)
            cc = self.kernel.dmmv(X, M, v=v_t, w=-Y, opt=opt)
            cc = prec.invTt(cc)
            reg = v.mul_(penalty * n)
            out = prec.invAt(cc.add_(reg))
            out = out.mul_(b / n)  # Rescale learning rate
            return out

    def solve(self,
              dset,
              nys_centers: torch.Tensor,
              penalty: float,
              n: int, m: int, t: int,
              initial_sol: Optional[torch.Tensor],
              max_iter: int,
              preconditioner: falkon.preconditioner.Preconditioner,
              opt: FalkonOptions,
              callback=None):
        if nys_centers is None:
            raise NotImplementedError("Pre-computed kernel is not supported by SGD.")
        cuda_inputs: bool = nys_centers.is_cuda
        device = nys_centers.device

        # Note that if we don't have CUDA this still works with stream=None.
        with ExitStack() as stack, TicToc("ConjGrad preparation", False), torch.inference_mode():
            if cuda_inputs:
                stream = torch.cuda.current_stream(device)
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(stream))

            g_step = functools.partial(self.sgd_iter,
                                       M=nys_centers, penalty=penalty, prec=preconditioner,
                                       n=n, opt=opt)
            if initial_sol is None:
                initial_sol = torch.zeros(m, t, dtype=nys_centers.dtype, device=device)

            # Run the conjugate gradient solver
            # noinspection PyTypeChecker
            beta = self.optimizer.solve(
                data=dset, X0=initial_sol, gradient_step=g_step, max_iter=max_iter,
                params=opt.get_gd_options(), callback=callback)

        return beta
