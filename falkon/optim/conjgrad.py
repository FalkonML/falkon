from contextlib import ExitStack
import functools
import time
from typing import Optional

import torch

import falkon
from falkon.options import ConjugateGradientOptions, FalkonOptions
from falkon.mmv_ops.fmmv_incore import incore_fdmmv, incore_fmmv
from falkon.utils.tensor_helpers import copy_same_stride, create_same_stride
from falkon.utils.stream_utils import get_non_default_stream
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
        Rsold = torch.sum(R.pow(2), dim=0)

        e_train = time.time() - t_start

        for i in range(max_iter):
            with TicToc("Chol Iter", debug=False):
                t_start = time.time()
                AP = mmv(P)
                alpha = Rsold / (torch.sum(P * AP, dim=0) + m_eps)
                X.addmm_(P, torch.diag(alpha))

                if (i + 1) % self.params.cg_full_gradient_every == 0:
                    R = B - mmv(X)
                else:
                    R = R - torch.mm(AP, torch.diag(alpha))
                    # R.addmm_(mat1=AP, mat2=torch.diag(alpha), alpha=-1.0)

                Rsnew = torch.sum(R.pow(2), dim=0)
                if Rsnew.abs().max().sqrt() < self.params.cg_tolerance:
                    print("Stopping conjugate gradient descent at "
                          "iteration %d. Solution has converged." % (i + 1))
                    break

                P = R + torch.mm(P, torch.diag(Rsnew / (Rsold + m_eps)))
                Rsold = Rsnew

                e_iter = time.time() - t_start
                e_train += e_iter
            with TicToc("Chol callback", debug=False):
                if callback is not None:
                    callback(i + 1, X, e_train)

        return X


class FalkonConjugateGradient(Optimizer):
    r"""Preconditioned conjugate gradient solver, optimized for the Falkon algorithm.

    The linear system solved is

    .. math::

        \widetilde{B}^\top H \widetilde{B} \beta = \widetilde{B}^\top K_{nm}^\top Y

    where :math:`\widetilde{B}` is the approximate preconditioner

    .. math::
        \widetilde{B} = 1/\sqrt{n}T^{-1}A^{-1}

    :math:`\beta` is the preconditioned solution vector (from which we can get :math:`\alpha = \widetilde{B}\beta`),
    and :math:`H` is the :math:`m\times m` sketched matrix

    .. math::
        H = K_{nm}^\top K_{nm} + \lambda n K_{mm}

    Parameters
    ----------
    kernel
        The kernel class used for the CG algorithm
    preconditioner
        The approximate Falkon preconditioner. The class should allow triangular solves with
        both :math:`T` and :math:`A` and multiple right-hand sides.
        The preconditioner should already have been initialized with a set of Nystrom centers.
        If the Nystrom centers used for CG are different from the ones used for the preconditioner,
        the CG method could converge very slowly.
    opt
        Options passed to the CG solver and to the kernel for computations.

    See Also
    --------
    :class:`falkon.preconditioner.FalkonPreconditioner`
        for the preconditioner class which is responsible for computing matrices `T` and `A`.
    """
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 preconditioner: falkon.preconditioner.Preconditioner,
                 opt: FalkonOptions,
                 weight_fn=None):
        super().__init__()
        self.kernel = kernel
        self.preconditioner = preconditioner
        self.params = opt
        self.optimizer = ConjugateGradient(opt.get_conjgrad_options())
        self.weight_fn = weight_fn

    def falkon_mmv(self, sol, penalty, X, M, Knm):
        n = Knm.shape[0] if Knm is not None else X.shape[0]
        prec = self.preconditioner

        with TicToc("MMV", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)

            if Knm is not None:
                cc = incore_fdmmv(Knm, v_t, None, opt=self.params)
            else:
                cc = self.kernel.dmmv(X, M, v_t, None, opt=self.params)

            # AT^-1 @ (TT^-1 @ (cc / n) + penalty * v)
            cc_ = cc.div_(n)
            v_ = v.mul_(penalty)
            cc_ = prec.invTt(cc_).add_(v_)
            out = prec.invAt(cc_)
            return out

    def weighted_falkon_mmv(self, sol, penalty, X, M, Knm, y_weights):
        n = Knm.shape[0] if Knm is not None else X.shape[0]
        prec = self.preconditioner

        with TicToc("MMV", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)

            if Knm is not None:
                cc = incore_fmmv(Knm, v_t, None, opt=self.params).mul_(y_weights)
                cc = incore_fmmv(Knm.T, cc, None, opt=self.params)
            else:
                cc = self.kernel.mmv(X, M, v_t, None, opt=self.params).mul_(y_weights)
                cc = self.kernel.mmv(M, X, cc, None, opt=self.params)

            # AT^-1 @ (TT^-1 @ (cc / n) + penalty * v)
            cc_ = cc.div_(n)
            v_ = v.mul_(penalty)
            cc_ = prec.invTt(cc_).add_(v_)
            out = prec.invAt(cc_)
            return out

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None):
        n = X.size(0)
        if M is None:
            Knm = X
        else:
            Knm = None

        cuda_inputs: bool = X.is_cuda
        device = X.device

        stream = None
        if cuda_inputs:
            stream = get_non_default_stream(device)

        # Note that if we don't have CUDA this still works with stream=None.
        with ExitStack() as stack, TicToc("ConjGrad preparation", False):
            if cuda_inputs:
                stack.enter_context(torch.cuda.device(device))
                stack.enter_context(torch.cuda.stream(stream))
            y_over_n = Y / n  # Cannot be in-place since Y needs to be preserved

            if self.is_weighted:
                y_weights = self.weight_fn(Y)
                y_over_n.mul_(y_weights)  # This can be in-place since we own y_over_n

            # Compute the right hand side
            if Knm is not None:
                B = incore_fmmv(Knm, y_over_n, None, transpose=True, opt=self.params)
            else:
                B = self.kernel.dmmv(X, M, None, y_over_n, opt=self.params)
            B = self.preconditioner.apply_t(B)

            if self.is_weighted:
                mmv = functools.partial(self.weighted_falkon_mmv, penalty=_lambda, X=X,
                                        M=M, Knm=Knm, y_weights=y_weights)
            else:
                mmv = functools.partial(self.falkon_mmv, penalty=_lambda, X=X, M=M, Knm=Knm)
            # Run the conjugate gradient solver
            beta = self.optimizer.solve(initial_solution, B, mmv, max_iter, callback)

        return beta

    @property
    def is_weighted(self):
        return self.weight_fn is not None
