import functools
import math
import time
from contextlib import ExitStack
from typing import Optional, Callable

import falkon
import torch

from falkon.options import GDOptions, FalkonOptions
from falkon.utils import TicToc
from falkon.mmv_ops.fmmv_incore import incore_fmmv
from falkon.optim import Optimizer, StopOptimizationException


class GradientDescent(Optimizer):
    def __init__(self):
        super().__init__()
        self.num_iter = None

    def solve(self,
              X0: torch.Tensor,
              gradient_step: Callable[[torch.Tensor], torch.Tensor],
              max_iter: int,
              learning_rate: float,
              params: GDOptions,
              callback: Optional[Callable[[int, torch.Tensor, float], None]] = None) -> torch.Tensor:
        t_start = time.time()

        tol = params.gd_tolerance ** 2
        e_train = time.time() - t_start
        for self.num_iter in range(max_iter):
            with TicToc("GD-Iter", debug=False):
                t_start = time.time()
                grad = gradient_step(X)
                X = X0 - learning_rate * grad

                diff = (X - X0).square().sum(dim=0).mean()
                if diff < tol:
                    break
                X0 = X
            e_iter = time.time() - t_start
            e_train += e_iter
            with TicToc("GD-callback", debug=False):
                if callback is not None:
                    try:
                        callback(self.num_iter + 1, X, e_train)
                    except StopOptimizationException as e:
                        print(f"Optimization stopped from callback: {e.message}")
                        break
        return X


class FalkonGradientDescent(Optimizer):
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 learning_rate: float,
                 weight_fn=None):
        super().__init__()
        if weight_fn is not None:
            raise NotImplementedError("Gradient solver does not support weight function. "
                                      "Use the CG solver instead.")
        self.kernel = kernel
        self.optimizer = GradientDescent()
        self.learning_rate = learning_rate

    def gd_iter(self, sol, penalty, X, M, Knm, Y, prec, opt):
        n = Knm.shape[0] if Knm is not None else X.shape[0]

        with TicToc("GD-ITER", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)
            if Knm is not None:
                cc = incore_fmmv(Knm, v_t, None, opt=opt)
            else:
                cc = self.kernel.mmv(X, M, v_t, None, opt=opt)
            cc.sub_(Y * math.sqrt(n))
            if Knm is not None:
                cc = incore_fmmv(Knm, cc, out=None, transpose=True, opt=opt)
            else:
                cc = self.kernel.mmv(M, X, cc, None, opt=opt)
            cc = prec.invTt(cc).div_(n)
            reg = v.mul_(penalty)
            out = prec.invAt(cc.add_(reg))
            return out

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter,
              preconditioner: falkon.preconditioner.Preconditioner, opt: FalkonOptions,
              callback=None):
        if M is None:
            Knm = X
            m = Knm.shape[1]
        else:
            Knm = None
            m = M.shape[0]

        cuda_inputs: bool = X.is_cuda
        device = X.device

        stream = None
        if cuda_inputs:
            stream = torch.cuda.current_stream(device)

        # Note that if we don't have CUDA this still works with stream=None.
        with ExitStack() as stack, TicToc("ConjGrad preparation", False), torch.inference_mode():
            if cuda_inputs:
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(stream))

            g_step = functools.partial(
                self.gd_iter, penalty=_lambda, X=X, M=M, Knm=Knm, Y=Y, prec=preconditioner)

            if initial_solution is None:
                initial_solution = torch.zeros(m, Y.shape[1], dtype=X.dtype, device=device)

            # Run the conjugate gradient solver
            beta = self.optimizer.solve(
                initial_solution, g_step, max_iter, learning_rate=self.learning_rate,
                params=opt.get_gd_options(), callback=callback)

        return beta
