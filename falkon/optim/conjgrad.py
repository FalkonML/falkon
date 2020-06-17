import time

import torch

from ..utils import CompOpt, TicToc


# function [x] = conjgrad(A, b, x)
#     r = b - A * x;
#     p = r;
#     rsold = r' * r;
#
#     for i = 1:length(b)
#         Ap = A * p;
#         alpha = rsold / (p' * Ap);
#         x = x + alpha * p;
#         r = r - alpha * Ap;
#         rsnew = r' * r;
#         if sqrt(rsnew) < 1e-10
#               break;
#         end
#         p = r + (rsnew / rsold) * p;
#         rsold = rsnew;
#     end
# end

class Optimizer(object):
    def __init__(self, opt=None, **kw):
        if opt is not None:
            self.params = CompOpt(opt)
        else:
            self.params = CompOpt()
        self.params.update(kw)


class ConjugateGradient(Optimizer):
    def __init__(self, opt=None, **kw):
        super().__init__(opt=opt, **kw)
        self.params.setdefault('cg_epsilon', {torch.float32: 1e-7, torch.float64: 1e-15})
        self.params.setdefault('cg_tolerance', 1e-10)
        self.params.setdefault('debug', False)
        self.params.setdefault('cg_full_gradient_every', 10)

    def solve(self, X0, B, mmv, max_iter, callback=None, opt=None):
        t_start = time.time()
        params = self.params.copy()
        if opt is not None:
            params = params.update(opt)

        if X0 is None:
            R = B.clone()
            X = torch.zeros_like(B)
        else:
            R = B - mmv(X0)
            X = X0

        if isinstance(params.cg_epsilon, dict):
            m_eps = params.cg_epsilon[X.dtype]
        else:
            m_eps = params.cg_epsilon

        P = R
        Rsold = torch.sum(R.pow(2), dim=0)

        e_train = time.time() - t_start

        for i in range(max_iter):
            with TicToc("Chol Iter", debug=params.debug):
                t_start = time.time()
                AP = mmv(P)
                alpha = Rsold / (torch.sum(P * AP, dim=0) + m_eps)
                X.addmm_(P, torch.diag(alpha))

                if (i + 1) % params.cg_full_gradient_every == 0:
                    R = B - mmv(X)
                else:
                    R = R - torch.mm(AP, torch.diag(alpha))
                    #R.addmm_(mat1=AP, mat2=torch.diag(alpha), alpha=-1.0)

                Rsnew = torch.sum(R.pow(2), dim=0)

                if Rsnew.abs().max().sqrt() < params.cg_tolerance:
                    print("Stopping conjugate gradient descent at "
                          "iteration %d. Solution has converged." % (i+1))
                    break

                P = R + torch.mm(P, torch.diag(Rsnew / (Rsold + m_eps)))
                Rsold = Rsnew

                e_iter = time.time() - t_start
                e_train += e_iter
            with TicToc("Chol callback", debug=False):#params.debug):
                if callback is not None:
                    callback(i + 1, X, e_train)

        return X


class FalkonConjugateGradient(ConjugateGradient):
    def __init__(self, kernel, preconditioner, opt=None, **kw):
        super().__init__(opt, **kw)
        self.params.setdefault('debug', False)
        self.kernel = kernel
        self.preconditioner = preconditioner

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None, opt=None):
        n = X.size(0)
        prec = self.preconditioner
        params = self.params.copy()
        if opt is not None:
            params = params.update(opt)

        with TicToc("ConjGrad preparation", False):# debug=params.debug):
            if M is None:
                Knm = X
            else:
                Knm = None
            # Compute the right hand side
            if Knm is not None:
                B = Knm.T @ (Y/n)
            else:
                B = self.kernel.dmmv(X, M, None, Y / n, opt=params.copy())
            B = prec.apply_t(B)

            # Define the Matrix-vector product iteration
            def mmv(sol):
                with TicToc("MMV", False):#debug=params.debug):
                    v = prec.invA(sol)
                    if Knm is not None:
                        cc = Knm.T @ (Knm @ prec.invT(v))
                    else:
                        cc = self.kernel.dmmv(X, M, prec.invT(v), None, opt=params.copy())
                    return prec.invAt(prec.invTt(cc / n) + _lambda * v)

        # Run the conjugate gradient solver
        beta = super().solve(initial_solution, B, mmv, max_iter, callback)
        return beta
