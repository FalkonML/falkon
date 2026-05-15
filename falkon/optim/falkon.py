import functools
from contextlib import ExitStack

import torch

import falkon
from falkon.optim.conjgrad import ConjugateGradient, Optimizer
from falkon.options import FalkonOptions
from falkon.utils import TicToc


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

    def __init__(
        self,
        kernel: falkon.kernels.Kernel,
        preconditioner: falkon.preconditioner.FalkonPreconditioner | falkon.preconditioner.LogisticPreconditioner,
        opt: FalkonOptions,
        weight_fn=None,
    ):
        super().__init__()
        self.kernel = kernel
        self.preconditioner = preconditioner
        self.params = opt
        self.optimizer = ConjugateGradient(opt.get_conjgrad_options())

        self.weight_fn = weight_fn

    def falkon_mmv(self, sol, penalty, X, M, n: int):
        prec = self.preconditioner

        with TicToc("MMV", False):
            with TicToc("Tri-solve 1", False):
                v = prec.invA(sol)
                v_t = prec.invT(v)
            with TicToc("DMMV", False):
                cc = self.kernel.dmmv(X, M, v_t, None, opt=self.params)

            with TicToc("Tri-solve 2", False):
                # AT^-1 @ (TT^-1 @ (cc / n) + penalty * v)
                cc_ = cc.div_(n)
                v_ = v.mul_(penalty)
                cc_ = prec.invTt(cc_).add_(v_)
                out = prec.invAt(cc_)
            return out

    def weighted_falkon_mmv(self, sol, penalty, X, M, y_weights, n: int):
        prec = self.preconditioner

        with TicToc("MMV", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)

            cc = self.kernel.mmv(X, M, v_t, None, opt=self.params).mul_(y_weights)
            cc = self.kernel.mmv(M, X, cc, None, opt=self.params)

            # AT^-1 @ (TT^-1 @ (cc / n) + penalty * v)
            cc_ = cc.div_(n)
            v_ = v.mul_(penalty)
            cc_ = prec.invTt(cc_).add_(v_)
            out = prec.invAt(cc_)
            return out

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter, callback=None):
        n = Y.size(0)
        cuda_inputs: bool = Y.is_cuda

        with ExitStack() as stack, TicToc("Falkon CG", False):
            if cuda_inputs:
                stream = torch.cuda.current_stream(Y.device)
                stack.enter_context(torch.cuda.device(Y.device))
                stack.enter_context(torch.cuda.stream(stream))
            y_over_n = Y / n  # Cannot be in-place since Y needs to be preserved

            # Compute the right hand side
            B = self.kernel.mmv(M, X, y_over_n, opt=self.params)
            B = self.preconditioner.apply_t(B)

            if self.is_weighted:
                assert self.weight_fn is not None
                y_weights = self.weight_fn(Y, X, torch.arange(Y.shape[0]))
                y_over_n.mul_(y_weights)  # This can be in-place since we own y_over_n
                mmv = functools.partial(self.weighted_falkon_mmv, penalty=_lambda, X=X, M=M, y_weights=y_weights, n=n)
            else:
                mmv = functools.partial(self.falkon_mmv, penalty=_lambda, X=X, M=M, n=n)
            # Run the conjugate gradient solver
            beta = self.optimizer.solve(initial_solution, B, mmv, max_iter, callback)

        return beta

    @property
    def is_weighted(self):
        return self.weight_fn is not None
