import time
from collections.abc import Callable

import torch

from falkon.options import ConjugateGradientOptions
from falkon.preconditioner.preconditioner import Preconditioner
from falkon.utils import TicToc
from falkon.utils.tensor_helpers import copy_same_stride, create_same_stride

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


class StopOptimizationException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


class Optimizer:
    """Base class for optimizers. This is an empty shell at the moment."""

    def __init__(self):
        pass


class PreconditionedConjugateGradient(Optimizer):
    def __init__(self, prec: Preconditioner, opt: ConjugateGradientOptions | None = None):
        super().__init__()
        self.params = opt or ConjugateGradientOptions()
        self.prec = prec
        self.num_iter = None

    def solve(
        self,
        x0: torch.Tensor | None,
        rhs: torch.Tensor,
        mmv: Callable[[torch.Tensor], torch.Tensor],
        max_iter: int,
        callback: Callable[[int, torch.Tensor, float], None] | None = None,
    ) -> torch.Tensor:
        T = rhs.shape[1]
        m_eps = self.params.cg_epsilon(rhs.dtype)
        full_grad_every = self.params.cg_full_gradient_every or max_iter + 1
        tol = (self.params.cg_tolerance * torch.linalg.vector_norm(rhs, dim=0)) ** 2

        # Differential convergence: when any column of X converges we remove it from optimization.
        diff_conv = self.params.cg_differential_convergence and T > 1
        # column-vectors of X which have converged
        x_converged: list[torch.Tensor] = []
        # indices of columns in `x_converged` as they originally appeared in `X`
        col_idx_converged: list[int] = []
        # indices of columns which have not converged, as they originally were in `X`
        col_idx_notconverged: torch.Tensor = torch.arange(T)

        with (timer := TicToc("PCG preparation", debug=False)):
            if x0 is None:
                r = copy_same_stride(rhs)  # n*T
                x = create_same_stride(rhs.size(), rhs, rhs.dtype, rhs.device)
                x.fill_(0.0)
            else:
                r = rhs - mmv(x0)  # n*T
                x = x0

            s = self.prec.apply(r)
            p = s  # no need to clone (unlike in conjgrad) since s gets modified.
            rs_old = (r * s).sum(dim=0)  # T
            x_orig = x  # keep a reference for differential convergence
            e_train = timer.toc_val()

        for self.num_iter in range(max_iter):
            with (timer := TicToc("Chol Iter", debug=False)):
                op_q = mmv(p)
                alpha = rs_old / (torch.sum(p * op_q, dim=0).add_(m_eps))
                # X += P @ diag(alpha)
                x.addcmul_(p, alpha.reshape(1, -1))

                if (self.num_iter + 1) % full_grad_every == 0:
                    if x.is_cuda:
                        # addcmul_ may not be finished yet causing mmv to get stale inputs.
                        torch.cuda.synchronize()
                    r = rhs - mmv(x)
                else:
                    # R -= AP @ diag(alpha)
                    r.addcmul_(op_q, alpha.reshape(1, -1), value=-1.0)

                s = self.prec.apply(r)
                rs_new = (r * s).sum(0)

                converged = torch.less(rs_new, tol)
                if torch.all(converged):
                    break
                if diff_conv and torch.any(converged):
                    for idx in torch.where(converged)[0]:
                        col_idx_converged.append(int(col_idx_notconverged[idx].item()))
                        x_converged.append(x[:, idx])
                    # These are all copies
                    col_idx_notconverged = col_idx_notconverged[~converged]
                    p = p[:, ~converged]
                    r = r[:, ~converged]
                    s = s[:, ~converged]
                    rhs = rhs[:, ~converged]
                    x = x[:, ~converged]
                    tol = tol[~converged]
                    rs_new = rs_new[~converged]
                    rs_old = rs_old[~converged]

                # P = R + P @ diag(mul)
                beta_multiplier = (rs_new / rs_old.add_(m_eps)).reshape(1, -1)
                p = p.mul_(beta_multiplier).add_(s)
                rs_old = rs_new
                e_train += timer.toc_val()
            with TicToc("Chol callback", debug=False):
                if callback is not None:
                    try:
                        callback(self.num_iter + 1, x, e_train)
                    except StopOptimizationException as e:
                        print(f"Optimization stopped from callback: {e.message}")
                        break
        if diff_conv:
            if len(x_converged) > 0:
                for i, out_idx in enumerate(col_idx_converged):
                    if x_orig[:, out_idx].data_ptr() != x_converged[i].data_ptr():
                        x_orig[:, out_idx].copy_(x_converged[i])
            if len(col_idx_notconverged) > 0:
                for i, out_idx in enumerate(col_idx_notconverged):
                    if x_orig[:, out_idx].data_ptr() != x[:, i].data_ptr():
                        x_orig[:, out_idx].copy_(x[:, i])
            x = x_orig
        return x


class ConjugateGradient(Optimizer):
    def __init__(self, opt: ConjugateGradientOptions | None = None):
        super().__init__()
        self.params = opt or ConjugateGradientOptions()
        self.num_iter = None

    def solve(
        self,
        X0: torch.Tensor | None,
        B: torch.Tensor,
        mmv: Callable[[torch.Tensor], torch.Tensor],
        max_iter: int,
        callback: Callable[[int, torch.Tensor, float], None] | None = None,
    ) -> torch.Tensor:
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

        m_eps = self.params.cg_epsilon(X.dtype)
        full_grad_every = self.params.cg_full_gradient_every or max_iter * 2
        tol = self.params.cg_tolerance**2
        diff_conv = self.params.cg_differential_convergence and X.shape[1] > 1

        P = R.clone()
        Rsold = R.square().sum(dim=0)

        e_train = time.time() - t_start

        # Differential convergence: when any column of X converges we remove it from optimization.
        # column-vectors of X which have converged
        x_converged: list[torch.Tensor] = []
        # indices of columns in `x_converged` as they originally appeared in `X`
        col_idx_converged: list[int] = []
        # indices of columns which have not converged, as they originally were in `X`
        col_idx_notconverged: torch.Tensor = torch.arange(X.shape[1])
        X_orig = X

        for self.num_iter in range(max_iter):
            with TicToc("Chol Iter", debug=False):
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
                        col_idx_converged.append(int(col_idx_notconverged[idx].item()))
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
            with TicToc("Chol callback", debug=False):
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
