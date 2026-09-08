import functools
from contextlib import ExitStack

import torch

import falkon
from falkon.optim.conjgrad import Optimizer, PreconditionedConjugateGradient
from falkon.options import FalkonOptions
from falkon.utils import TicToc


class BalkonConjugateGradient(Optimizer):
    def __init__(
        self,
        kernel: falkon.kernels.Kernel,
        preconditioner: falkon.preconditioner.BalkonPreconditioner,
        opt: FalkonOptions,
    ):
        super().__init__()
        self.kernel = kernel
        self.preconditioner = preconditioner
        self.params = opt
        self.optimizer = PreconditionedConjugateGradient(preconditioner, opt.get_conjgrad_options())

    def balkon_mmv(self, sol, penalty, X, M, n: int):
        with TicToc("MMV", False):
            KMMv = self.kernel.mmv(M, M, sol, opt=self.params)
            KMMv.mul_(penalty / self.preconditioner.num_blocks)

            dKMMv = self.kernel.dmmv(X, M, sol, None, opt=self.params)
            dKMMv.div_(n * self.preconditioner.num_blocks)

            dKMMv.add_(KMMv)
            return dKMMv

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None):
        n = Y.size(0)
        cuda_inputs: bool = Y.is_cuda
        device = Y.device

        with ExitStack() as stack, TicToc("ConjGrad preparation", False):
            if cuda_inputs:
                stream = torch.cuda.current_stream(device)
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(stream))

            # Compute the right hand side
            y_over_n = Y / (n * self.preconditioner.num_blocks)  # cannot modify Y inplace.
            B = self.kernel.mmv(M, X, y_over_n, opt=self.params)

            mmv = functools.partial(self.balkon_mmv, penalty=_lambda, X=X, M=M, n=n)
            alpha = self.optimizer.solve(initial_solution, B, mmv, max_iter, callback)

        return alpha
