import functools
import time
from contextlib import ExitStack
from typing import Optional, Callable, List

import torch

import falkon
from falkon.mmv_ops.fmmv_incore import incore_fdmmv, incore_fmmv
from falkon.options import ConjugateGradientOptions, FalkonOptions
from falkon.utils import TicToc
from falkon.utils.tensor_helpers import copy_same_stride, create_same_stride
from optim import Optimizer, StopOptimizationException


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


class ConjugateGradient(Optimizer):
    def __init__(self):
        super().__init__()
        self.num_iter = None

    def solve(self,
              X0: Optional[torch.Tensor],
              B: torch.Tensor,
              mmv: Callable[[torch.Tensor], torch.Tensor],
              max_iter: int,
              params: ConjugateGradientOptions,
              callback: Optional[Callable[[int, torch.Tensor, float], None]] = None) -> torch.Tensor:
        """Conjugate-gradient solver with optional support for preconditioning via generic MMV.

        This solver can be used for iterative solution of linear systems of the form $AX = B$ with
        respect to the `X` variable. Knowledge of `A` is only needed through matrix-vector
        multiplications with temporary solutions (must be provided through the `mmv` function).

        Preconditioning can be achieved by incorporating the preconditioner matrix in the `mmv`
        function.

        Parameters
        ----------
        X0 : Optional[torch.Tensor]
            Initial solution for the solver. If not provided it will be a zero-tensor.
        B : torch.Tensor
            Right-hand-side of the linear system to be solved.
        mmv
            User-provided function to perform matrix-vector multiplications with the design matrix
            `A`. The function must accept a single argument (the vector to be multiplied), and
            return the result of the matrix-vector multiplication.
        max_iter : int
            Maximum number of iterations the solver will perform. Early stopping is implemented
            via the options passed in the constructor of this class (in particular look at
            `cg_tolerance` options)
            i + 1, X, e_train
        callback
            An optional, user-provided function which shall be called at the end of each iteration
            with the current solution. The arguments to the function are the iteration number,
            a tensor containing the current solution, and the total time elapsed from the beginning
            of training (note that this time explicitly excludes any time taken by the callback
            itself).
        Returns
        -------
        The solution to the linear system `X`.
        """
        t_start = time.time()

        if X0 is None:
            R = copy_same_stride(B)  # n*t
            X = create_same_stride(B.size(), B, B.dtype, B.device)
            X.fill_(0.0)
        else:
            R = B - mmv(X0)  # n*t
            X = X0

        m_eps = params.cg_epsilon(X.dtype)
        full_grad_every = params.cg_full_gradient_every or max_iter * 2
        tol = params.cg_tolerance ** 2
        diff_conv = params.cg_differential_convergence and X.shape[1] > 1

        P = R.clone()
        R0 = R.square().sum(dim=0)
        Rsold = R0.clone()

        e_train = time.time() - t_start

        if diff_conv:
            # Differential convergence: when any column of X converges we remove it from optimization.
            # column-vectors of X which have converged
            x_converged: List[torch.Tensor] = []
            # indices of columns in `x_converged` as they originally appeared in `X`
            col_idx_converged: List[int] = []
            # indices of columns which have not converged, as they originally were in `X`
            col_idx_notconverged: torch.Tensor = torch.arange(X.shape[1])
            X_orig = X

        for self.num_iter in range(max_iter):
            with TicToc("CG-Iter", debug=False):
                t_start = time.time()
                AP = mmv(P)
                alpha = Rsold / (torch.sum(P * AP, dim=0).add_(m_eps))
                # X += P @ diag(alpha)
                X.addcmul_(P, alpha.reshape(1, -1))

                if (self.num_iter + 1) % full_grad_every == 0:
                    if X.is_cuda:
                        # addmm_ may not be finished yet causing mmv to get stale inputs.
                        torch.cuda.synchronize()
                    R = B - mmv(X)
                else:
                    # R -= AP @ diag(alpha)
                    R.addcmul_(AP, alpha.reshape(1, -1), value=-1.0)

                Rsnew = R.square().sum(dim=0)  # t
                converged = torch.less(Rsnew, tol)
                if torch.all(converged):
                    break
                if diff_conv and torch.any(converged):
                    for idx in torch.where(converged)[0]:
                        col_idx_converged.append(col_idx_notconverged[idx])
                        x_converged.append(X[:, idx])
                    col_idx_notconverged = col_idx_notconverged[~converged]
                    P = P[:, ~converged]
                    R = R[:, ~converged]
                    B = B[:, ~converged]
                    X = X[:, ~converged]  # These are all copies
                    Rsnew = Rsnew[~converged]
                    Rsold = Rsold[~converged]

                # P = R + P @ diag(mul)
                multiplier = (Rsnew / Rsold.add_(m_eps)).reshape(1, -1)
                P = P.mul_(multiplier).add_(R)
                Rsold = Rsnew

                e_iter = time.time() - t_start
                e_train += e_iter
            with TicToc("CG-callback", debug=False):
                if callback is not None:
                    try:
                        callback(self.num_iter + 1, X, e_train)
                    except StopOptimizationException as e:
                        print(f"Optimization stopped from callback: {e.message}")
                        break
        if diff_conv:
            if len(x_converged) > 0:
                for i, out_idx in enumerate(col_idx_converged):
                    if X_orig[:, out_idx].data_ptr() != x_converged[i].data_ptr():
                        X_orig[:, out_idx].copy_(x_converged[i])
            if len(col_idx_notconverged) > 0:
                for i, out_idx in enumerate(col_idx_notconverged):
                    if X_orig[:, out_idx].data_ptr() != X[:, i].data_ptr():
                        X_orig[:, out_idx].copy_(X[:, i])
            X = X_orig
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
                 weight_fn=None):
        super().__init__()
        self.kernel = kernel
        self.optimizer = ConjugateGradient()
        self.weight_fn = weight_fn

    def falkon_mmv(self, sol, penalty, X, M, Knm, prec, opt):
        n = Knm.shape[0] if Knm is not None else X.shape[0]

        with TicToc("MMV", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)

            if Knm is not None:
                cc = incore_fdmmv(Knm, v_t, None, opt=opt)
            else:
                cc = self.kernel.dmmv(X, M, v_t, None, opt=opt)

            # AT^-1 @ (TT^-1 @ (cc / n) + penalty * v)
            cc_ = cc.div_(n)
            v_ = v.mul_(penalty)
            cc_ = prec.invTt(cc_).add_(v_)
            out = prec.invAt(cc_)
            return out

    def weighted_falkon_mmv(self, sol, penalty, X, M, Knm, y_weights, prec, opt):
        n = Knm.shape[0] if Knm is not None else X.shape[0]

        with TicToc("MMV", False):
            v = prec.invA(sol)
            v_t = prec.invT(v)

            if Knm is not None:
                cc = incore_fmmv(Knm, v_t, None, opt=opt).mul_(y_weights)
                cc = incore_fmmv(Knm.T, cc, None, opt=opt)
            else:
                cc = self.kernel.mmv(X, M, v_t, None, opt=opt).mul_(y_weights)
                cc = self.kernel.mmv(M, X, cc, None, opt=opt)

            # AT^-1 @ (TT^-1 @ (cc / n) + penalty * v)
            cc_ = cc.div_(n)
            v_ = v.mul_(penalty)
            cc_ = prec.invTt(cc_).add_(v_)
            out = prec.invAt(cc_)
            return out

    def solve(self, X, M, Y, _lambda, initial_solution, max_iter,
              preconditioner: falkon.preconditioner.Preconditioner, opt: FalkonOptions,
              callback=None):
        n = X.size(0)
        if M is None:
            Knm = X
        else:
            Knm = None

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
            y_over_n = Y / n  # Cannot be in-place since Y needs to be preserved

            if self.is_weighted:
                y_weights = self.weight_fn(Y, X, torch.arange(Y.shape[0]))
                y_over_n.mul_(y_weights)  # This can be in-place since we own y_over_n

            # Compute the right hand side
            if Knm is not None:
                B = incore_fmmv(Knm, y_over_n, None, transpose=True, opt=opt)
            else:
                B = self.kernel.mmv(M, X, y_over_n, opt=opt)
            B = preconditioner.apply_t(B)

            if self.is_weighted:
                mmv = functools.partial(self.weighted_falkon_mmv, penalty=_lambda, X=X,
                                        M=M, Knm=Knm, y_weights=y_weights, prec=preconditioner,
                                        opt=opt)
            else:
                mmv = functools.partial(self.falkon_mmv, penalty=_lambda, X=X, M=M, Knm=Knm,
                                        prec=preconditioner, opt=opt)
            # Run the conjugate gradient solver
            beta = self.optimizer.solve(initial_solution, B, mmv, max_iter=max_iter,
                                        params=opt.get_conjgrad_options(), callback=callback)

        return beta

    @property
    def is_weighted(self):
        return self.weight_fn is not None
