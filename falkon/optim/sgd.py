import math
import time

import torch
import numpy as np
from ..utils import CompOpt
from . import Optimizer


class SGD(Optimizer):
    def __init__(self, opt=None, **kw):
        super().__init__(opt=opt, **kw)
        self.params.setdefault('opt_tolerance', 1e-10)
        self.params.setdefault('mb_size', 128)
        self.params.setdefault('step_size', '1/sqrt')

    def solve(self, X0, mmv, num_points, max_iter, callback=None):
        if self.params.step_size == '1/sqrt':
            step_size = min(1, 1 / math.sqrt(num_points))
        else:
            step_size = float(self.params.step_size)
        mb_size = self.params.mb_size
        X = X0

        e_train = 0
        t_start = time.time()

        per_epoch_iters = int(math.ceil(num_points / mb_size))
        for i in range(max_iter):
            for j in range(per_epoch_iters):
                mb_idx = torch.randint(num_points, (mb_size, ), dtype=torch.long)
                X = X - step_size * mmv(X, mb_idx)

            e_iter = time.time() - t_start
            e_train += e_iter
            if callback is not None:
                callback(i + 1, X, e_iter, e_train)

        return X


class FalkonSGD(SGD):
    def __init__(self, kernel, preconditioner, opt=None, **kw):
        super().__init__(opt, **kw)
        self.params.setdefault('inter_type', torch.float64)
        self.kernel = kernel
        self.preconditioner = preconditioner

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None):
        n = X.shape[0]
        prec = self.preconditioner

        # Compute the right hand side
        B = prec.apply_t(self.kernel.dmmv(
                    X, M, None, Y / n, opt=self.params.copy()))

        X0 = initial_solution
        if initial_solution is None:
            X0 = torch.zeros((M.shape[0], Y.shape[1]), dtype=self.params.inter_type)

        # Define the Matrix-vector product iteration
        def mb_mmv(sol, mb_idx):
            v = prec.invA(sol)
            cc = self.kernel.dmmv(X[mb_idx, :], M, prec.invT(v), -Y[mb_idx, :],
                                  opt=self.params.copy())
            return prec.invAt(prec.invTt(cc / n) + _lambda * v)

        # Run the conjugate gradient solver
        beta = super().solve(X0, mb_mmv, n, max_iter, callback)
        return beta
