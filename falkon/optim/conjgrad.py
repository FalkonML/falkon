import time
from typing import Optional

import torch

from falkon.options import ConjugateGradientOptions, FalkonOptions
from falkon.mmv_ops.fmmv_incore import incore_fdmmv, incore_fmmv
from falkon.utils.tensor_helpers import copy_same_stride, create_same_stride
from falkon.utils import TicToc

# More readable 'pseudocode' for conjugate gradient.
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
    def __init__(self):
        pass


class ConjugateGradient(Optimizer):
    def __init__(self, opt: Optional[ConjugateGradientOptions] = None):
        super().__init__()
        self.params = opt or ConjugateGradientOptions()

    def solve(self, X0, B, mmv, max_iter, callback=None):
        t_start = time.time()

        if X0 is None:
            R = copy_same_stride(B)
            X = create_same_stride(B.size(), B, B.dtype, B.device)
            X.fill_(0.0)
        else:
            R = B - mmv(X0)
            X = X0

        m_eps = self.params.cg_epsilon(X.dtype)

        P = R
        # noinspection PyArgumentList
        Rsold = torch.sum(R.pow(2), dim=0)

        e_train = time.time() - t_start

        for i in range(max_iter):
            with TicToc("Chol Iter", debug=False):  # TODO: FIXME
                t_start = time.time()
                AP = mmv(P)
                # noinspection PyArgumentList
                alpha = Rsold / (torch.sum(P * AP, dim=0) + m_eps)
                X.addmm_(P, torch.diag(alpha))

                if (i + 1) % self.params.cg_full_gradient_every == 0:
                    R = B - mmv(X)
                else:
                    R = R - torch.mm(AP, torch.diag(alpha))
                    # R.addmm_(mat1=AP, mat2=torch.diag(alpha), alpha=-1.0)

                # noinspection PyArgumentList
                Rsnew = torch.sum(R.pow(2), dim=0)
                if Rsnew.abs().max().sqrt() < self.params.cg_tolerance:
                    print("Stopping conjugate gradient descent at "
                          "iteration %d. Solution has converged." % (i + 1))
                    break

                P = R + torch.mm(P, torch.diag(Rsnew / (Rsold + m_eps)))
                if P.is_cuda:
                    # P must be synced so that it's correct for mmv in next iter.
                    torch.cuda.synchronize()
                Rsold = Rsnew

                e_iter = time.time() - t_start
                e_train += e_iter
            with TicToc("Chol callback", debug=False):
                if callback is not None:
                    callback(i + 1, X, e_train)

        return X


class FalkonConjugateGradient(Optimizer):
    def __init__(self, kernel, preconditioner, opt: FalkonOptions):
        super().__init__()
        self.kernel = kernel
        self.preconditioner = preconditioner
        self.params = opt
        self.optimizer = ConjugateGradient(opt.get_conjgrad_options())

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None):
        n = X.size(0)
        prec = self.preconditioner

        with TicToc("ConjGrad preparation", False):
            if M is None:
                Knm = X
            else:
                Knm = None
            # Compute the right hand side
            if Knm is not None:
                B = incore_fmmv(Knm, Y / n, None, transpose=True, opt=self.params)
            else:
                B = self.kernel.dmmv(X, M, None, Y / n, opt=self.params)

            B = prec.apply_t(B)

            # Define the Matrix-vector product iteration
            if X.is_cuda:
                s1 = torch.cuda.Stream(X.device)

            def mmv(sol):
                with TicToc("MMV", False):
                    v = prec.invA(sol)
                    v_t = prec.invT(v)
                    if Knm is not None:
                        cc = incore_fdmmv(Knm, v_t, None, opt=self.params)
                    else:
                        cc = self.kernel.dmmv(X, M, v_t, None, opt=self.params)

                    if X.is_cuda:
                        with torch.cuda.stream(s1):
                            cc_ = cc.div_(n)
                            v_ = v.mul_(_lambda)
                            s1.synchronize()
                            cc_ = prec.invTt(cc_).add_(v_)
                            s1.synchronize()
                            out = prec.invAt(cc_)
                            s1.synchronize()
                            return out
                    else:
                        return prec.invAt(prec.invTt(cc / n) + _lambda * v)

        # Run the conjugate gradient solver
        beta = self.optimizer.solve(initial_solution, B, mmv, max_iter, callback)
        return beta
