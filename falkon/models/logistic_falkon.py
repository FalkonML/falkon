import time
from typing import Union, Optional, List, Callable, Tuple

import torch

import falkon
from falkon.gsc_losses import Loss
from falkon.models.model_utils import FalkonBase
from falkon.optim import ConjugateGradient
from falkon.options import *
from falkon.utils import TicToc

__all__ = ("LogisticFalkon", )


class LogisticFalkon(FalkonBase):
    """Falkon Logistic regression solver.

    This estimator object solves approximate logistic regression problems with Nystroem
    projections and a fast optimization algorithm as described in  :ref:`[1] <flk_1>`, :ref:`[3] <log_flk>`.

    This model can handle logistic regression, so it may be used in place of
    :class:`falkon.models.Falkon` (which uses the squared loss) when tackling binary
    classification problems.

    The algorithm works by repeated applications of the base falkon algorithm with decreasing
    amounts of regularization; therefore the class accepts slightly different parameters from
    :class:`falkon.models.Falkon`: a `penalty_list` which should contain a list of decreasing
    regularization amounts, and an `iter_list` which should specify for each application
    of the base algorithm, how many CG iterations to use. For guidance on how to set these
    parameters, see below.

    Parameters
    -----------
    kernel
        Object representing the kernel function used for KRR.
    penalty_list : List[float]
        Amount of regularization to use for each iteration of the base algorithm. The length
        of this list determines the number of base algorithm iterations.
    iter_list: List[int]
        Number of conjugate gradient iterations used in each iteration of the base algorithm.
        The length of this list must be identical to that of `penalty_list`.
    loss: Loss
        This parameter must be set to an instance of :class:`falkon.gsc_losses.LogisticLoss`,
        initialized with the same kernel as this class.
    M : int
        The number of Nystrom centers to pick. `M` must be positive,
        and lower than the total number of training points. A larger
        `M` will typically lead to better accuracy but will use more
        computational resources.
    center_selection : str or falkon.center_selection.CenterSelector
        The center selection algorithm. Implemented is only 'uniform'
        selection which can choose each training sample with the same
        probability.
    seed : int or None
        Random seed. Can be used to make results stable across runs.
        Randomness is present in the center selection algorithm, and in
        certain optimizers.
    error_fn : Callable or None
        A function with two arguments: targets and predictions, both :class:`torch.Tensor`
        objects which returns the error incurred for predicting 'predictions' instead of
        'targets'. This is used to display the evolution of the error during the iterations.
    error_every : int or None
        Evaluate the error (on training or validation data) every
        `error_every` iterations. If set to 1 then the error will be
        calculated at each iteration. If set to None, it will never be
        calculated.
    options : FalkonOptions
        Additional options used by the components of the Falkon solver. Individual options
        are documented in :mod:`falkon.options`.

    Examples
    --------
    Running Logistic Falkon on a random dataset

    >>> X = torch.randn(1000, 10)
    >>> Y = torch.randn(1000, 1)
    >>> Y[Y > 0] = 1
    >>> Y[Y <= 0] = -1
    >>> kernel = falkon.kernels.GaussianKernel(3.0)
    >>> options = FalkonOptions()
    >>> model = LogisticFalkon(kernel=kernel, penalty_list=[1e-2, 1e-4, 1e-6, 1e-6, 1e-6],
    >>>                        iter_list=[3, 3, 3, 8, 8], M=500, options=options)
    >>> model.fit(X, Y)
    >>> preds = model.predict(X)

    References
    ----------
     - Ulysse Marteau-Ferey, Francis Bach, Alessandro Rudi, "Globally Convergent Newton Methods
       for Ill-conditioned Generalized Self-concordant Losses," NeurIPS 32, 2019.
     - Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi,
       "Kernel methods through the roof: handling billions of points efficiently,"
       Advancs in Neural Information Processing Systems, 2020.

    Notes
    -----
    A rule of thumb for **setting the `penalty_list`** is to keep in mind the desired final
    regularization (1e-6 in the example above), and then create a short path of around three
    iterations where the regularization is decreased down to the desired value. The decrease can
    be of 10^2 or 10^3 at each step. Then a certain number of iterations at the desired
    regularization may be necessary to achieve good performance.
    The `iter_list` attribute  follows a similar reasoning: use 3 inner-steps for the first three
    iterations where the regularization is decreased, and then switch to a higher number of
    inner-steps (e.g. 8) for the remaining iterations.
    """
    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty_list: List[float],
                 iter_list: List[int],
                 loss: Loss,
                 M: int,
                 center_selection: Union[str, falkon.center_selection.CenterSelector] = 'uniform',
                 seed: Optional[int] = None,
                 error_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Union[float, Tuple[float, str]]]] = None,
                 error_every: Optional[int] = 1,
                 options: Optional[FalkonOptions] = None,
                 ):
        super().__init__(kernel, M, center_selection, seed, error_fn, error_every, options)
        self.penalty_list = penalty_list
        self.iter_list = iter_list
        if len(self.iter_list) != len(self.penalty_list):
            raise ValueError("Iteration list must be of same length as penalty list "
                             "(found %d and %d)" % (len(self.iter_list), len(self.penalty_list)))
        self.loss = loss
        self._init_cuda()

    def fit(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            Xts: Optional[torch.Tensor] = None,
            Yts: Optional[torch.Tensor] = None):
        """Fits the Falkon Kernel Logistic Regression model.

        Parameters
        -----------
        X : torch.Tensor
            The tensor of training data, of shape [num_samples, num_dimensions].
            If X is in Fortran order (i.e. column-contiguous) then we can avoid
            an extra copy of the data.
        Y : torch.Tensor
            The tensor of training targets, of shape [num_samples, num_outputs].
            If X and Y represent a classification problem, Y can be encoded as a one-hot
            vector.
            If Y is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.
        Xts : torch.Tensor or None
            Tensor of validation data, of shape [num_test_samples, num_dimensions].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Xts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.
        Yts : torch.Tensor or None
            Tensor of validation targets, of shape [num_test_samples, num_outputs].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Yts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.

        Returns
        --------
        model: LogisticFalkon
            The fitted model
        """
        X, Y, Xts, Yts = self._check_fit_inputs(X, Y, Xts, Yts)
        if Y.shape[1] != 1:
            raise ValueError("Logistic Falkon expects a single response variable "
                             "(Y must be 1D or the second dimension must be of size 1).")

        dtype = X.dtype
        # Add a dummy `fit_time` to make compatible with normal falkon
        # where the first `fit_time` is just preparation time.
        self.fit_times_ = [0.0]

        t_s = time.time()

        with torch.autograd.inference_mode():
            ny_X, ny_Y = self.center_selection.select(X, Y)
            if self.use_cuda_:
                ny_X = ny_X.pin_memory()
                ny_Y = ny_Y.pin_memory()

            beta_it = torch.zeros(ny_X.shape[0], 1, dtype=dtype)  # Temporary iterative solution
            optim = ConjugateGradient()
            precond = falkon.preconditioner.LogisticPreconditioner(self.kernel, self.loss, self.options)

            # Define the validation callback TODO: this is almost identical to the generic cback in model_utils.py
            validation_cback = None
            if self.error_fn is not None and self.error_every is not None:
                def validation_cback(it, beta, train_time):
                    self.fit_times_.append(self.fit_times_[0] + train_time)
                    if it % self.error_every != 0:
                        print(f"Iteration {it:3d} - Elapsed {self.fit_times_[-1]:.2f}s", flush=True)
                        return
                    err_str = "training" if Xts is None or Yts is None else "validation"
                    coeff = self._params_to_original_space(beta, precond)
                    # Compute error: can be train or test
                    if Xts is not None and Yts is not None:
                        pred = self._predict(Xts, ny_X, coeff)
                        err = self.error_fn(Yts, pred)
                        loss = torch.mean(self.loss(Yts, pred)).item()
                    else:
                        pred = self._predict(X, ny_X, coeff)
                        err = self.error_fn(Y, pred)
                        loss = torch.mean(self.loss(Y, pred)).item()
                    err_name = "error"
                    if isinstance(err, tuple) and len(err) == 2:
                        err, err_name = err
                    print(f"Iteration {it:3d} - Elapsed {self.fit_times_[-1]:.2f}s - "
                          f"{err_str} loss {loss:.4f} - "
                          f"{err_str} {err_name} {err:.4f} ", flush=True)

            # Start iterative training procedure:
            # each iteration computes preconditioner and falkon iterations.
            t_elapsed = 0.0
            for out_iter, penalty in enumerate(self.penalty_list):
                max_iter = self.iter_list[out_iter]
                print("Iteration %d - penalty %e - sub-iterations %d" % (out_iter, penalty, max_iter), flush=True)

                with TicToc("Preconditioner", self.options.debug):
                    precond.init(ny_X, ny_Y, beta_it, penalty, X.shape[0])

                if self.use_cuda_:  # TODO: Test if this is necessary
                    torch.cuda.empty_cache()

                with TicToc("Gradient", self.options.debug):
                    # Gradient computation
                    knmp_grad, inner_mmv = self.loss.knmp_grad(
                        X, ny_X, Y, precond.invT(beta_it), opt=self.options)
                    grad_p = precond.invAt(precond.invTt(knmp_grad).add_(penalty * beta_it))

                # Run CG with `grad_p` as right-hand-side.
                with TicToc("Optim", self.options.debug):
                    def mmv(sol):
                        sol_a = precond.invA(sol)
                        knmp_hess = self.loss.knmp_hess(
                            X, ny_X, Y, inner_mmv, precond.invT(sol_a), opt=self.options)
                        return precond.invAt(precond.invTt(knmp_hess).add_(sol_a.mul_(penalty)))
                    optim_out = optim.solve(X0=None, B=grad_p, mmv=mmv, max_iter=max_iter,
                                            params=self.options)
                    beta_it -= precond.invA(optim_out)

                t_elapsed += time.time() - t_s
                if validation_cback is not None:
                    validation_cback(out_iter, beta_it, train_time=t_elapsed)
                t_s = time.time()
            t_elapsed += time.time() - t_s

            self.alpha_ = precond.invT(beta_it)
            self.ny_points_ = ny_X
        return self

    def _predict(self, X, ny_points, alpha):
        with torch.autograd.inference_mode():
            return self.kernel.mmv(X, ny_points, alpha, opt=self.options)

    def _params_to_original_space(self, params, preconditioner):
        return preconditioner.invT(params)
