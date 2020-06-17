import numbers
import time
from typing import Union, Optional

import numpy as np
import torch
from sklearn import base

import falkon
from falkon.ooc_ops.options import LauumOptions, CholeskyOptions
from falkon.utils import CompOpt, TicToc, devices
from falkon.utils.devices import get_device_info
from falkon.utils.helpers import decide_cuda, sizeof_dtype, check_same_dtype

__all__ = ("Falkon",)


def get_min_cuda_preconditioner_size(dt):
    if dt == torch.float32:
        return 10_000
    else:
        return 30_000


def get_min_cuda_mmv_size(dt):
    if dt == torch.float32:
        return 10_000 * 10 * 3_000
    else:
        return 30_000 * 10 * 3_000


def check_random_generator(seed):
    """Turn seed into a np.random.Generator instance

    Parameters
    ----------
    seed : None | int | instance of Generator
        If seed is None, return the Generator singleton used by np.random.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class Falkon(base.BaseEstimator):
    """FALKON Kernel Ridge Regression solver

    FALKON solves KRR problems with the squared loss in an approximate
    way using a Nystrom projection and a fast optimization algorithm.

    Multiclass and multiple regression problems can all be tackled
    with this same object, for example by encoding multiple classes
    in a one-hot target matrix.

    Parameters:
    -----------
     - kernel : falkon.Kernel
        Object representing the kernel function used for KRR.
     - penalty : float
        Amount of regularization to apply to the problem.
        This parameter must be greater than 0.
     - M : int
        The number of Nystrom centers to pick. `M` must be positive,
        and lower than the total number of training points. A larger
        `M` will typically lead to better accuracy but will use more
        computational resources.
     - precond : str or falkon.precond.Preconditioner
        The preconditioner to be used. The default is to use the 'falkon'
        preconditioner which leads to fast and accurate results
     - optim : str or falkon.optim.Optimizer
        The optimization algorithm to be used. The default is to use
        the conjugate gradient optimization algorithm ('conjgrad')
     - center_selection : str or falkon.center_selection.NySel
        The center selection algorithm. Implemented is only 'uniform'
        selection which can choose each training sample with the same
        probability.
     - maxiter : int
        The number of iterations to run the optimization for. Usually
        fewer than 20 iterations are necessary, however this is problem
        dependent.
     - seed : int or None
        Random seed. Can be used to make results stable across runs.
        Randomness is present in the center selection algorithm, and in
        certain optimizers.
     - error_fn : callable or None
        A function which can be called with targets and predictions as
        arguments and returns the error of the predictions. This is used
        to display the evolution of the error during the iterations.
     - error_every : int or None
        Evaluate the error (on training or validation data) every
        `error_every` iterations. If set to 1 then the error will be
        calculated at each iteration. If set to None, it will never be
        calculated.

    Notes:
    ------
    Additional parameters are documented here.
     - TODO: document
    """

    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty: float,
                 M: int,
                 center_selection: Union[str, falkon.center_selection.NySel] = 'uniform',
                 maxiter: int = 20,
                 seed: Optional[int] = None,
                 error_fn: Optional[callable] = None,
                 error_every: Optional[int] = 1,
                 use_cpu=False,
                 compute_arch_speed=False,
                 max_cpu_mem=np.inf,
                 max_gpu_mem=np.inf,
                 cholesky_opt=CholeskyOptions(),
                 lauum_opt=LauumOptions(),
                 pc_epsilon=None,
                 cpu_preconditioner=False,
                 cg_tolerance=1e-7,
                 no_keops=False,
                 no_single_kernel=True,  # TODO: Rename
                 debug=True,
                 ):
        self.kernel = kernel
        self.penalty = penalty
        self.M = M
        self.maxiter = maxiter
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)  # Works for both CPU and GPU
        self.random_state_ = check_random_generator(self.seed)

        self.error_fn = error_fn
        self.error_every = error_every
        # Options
        self.use_cpu = use_cpu or not torch.cuda.is_available()
        self.compute_arch_speed = compute_arch_speed
        self.max_gpu_mem = max_gpu_mem
        self.max_cpu_mem = max_cpu_mem
        self.pc_epsilon = pc_epsilon
        self.cholesky_opt = cholesky_opt
        self.lauum_opt = lauum_opt
        if pc_epsilon is None:
            self.pc_epsilon = {torch.float32: 1e-6, torch.float64: 1e-13}
        self.cpu_preconditioner = cpu_preconditioner
        self.cg_tolerance = cg_tolerance
        self.no_keops = no_keops
        self.no_single_kernel = no_single_kernel
        self.debug = debug

        self.extra_opt_ = CompOpt({
            'use_cpu': self.use_cpu,
            'compute_arch_speed': self.compute_arch_speed, 'max_cpu_mem': self.max_cpu_mem,
            'max_gpu_mem': self.max_gpu_mem,
            'cholesky_opt': cholesky_opt, 'lauum_opt': lauum_opt,
            'pc_epsilon': self.pc_epsilon,
            'cpu_preconditioner': self.cpu_preconditioner, 'cg_tolerance': self.cg_tolerance,
            'no_keops': self.no_keops,
            'no_single_kernel': self.no_single_kernel, 'debug': self.debug,
        })

        self.use_cuda_ = decide_cuda(self.extra_opt_)
        self.alpha_ = None
        self.ny_points_ = None
        self.fit_times_ = None

        if isinstance(center_selection, str):
            if center_selection.lower() == 'uniform':
                self.center_selection = falkon.center_selection.UniformSel(
                    self.random_state_, self.extra_opt_)
            else:
                raise ValueError(f'Center selection "{center_selection}" is not valid.')
        else:
            self.center_selection = center_selection

        self._init_cuda()

    def _init_cuda(self):
        if self.use_cuda_:
            torch.cuda.init()
            from falkon.cuda import initialization
            initialization.init(self.extra_opt_)
            self.num_gpus = devices.num_gpus(self.extra_opt_)

    def fit(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            Xts: Optional[torch.Tensor] = None,
            Yts: Optional[torch.Tensor] = None):
        """Fits the Falkon KRR model.

        Parameters:
        -----------
         - X : torch.Tensor (2D)
            The tensor of training data, of shape [num_samples, num_dimensions].
            If X is in Fortran order (i.e. column-contiguous) then we can avoid
            an extra copy of the data.
         - Y : torch.Tensor (1D or 2D)
            The tensor of training targets, of shape [num_samples, num_outputs].
            If X and Y represent a classification problem, Y can be encoded as a one-hot
            vector.
            If Y is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.
         - Xts : torch.Tensor (2D) or None
            Tensor of validation data, of shape [num_test_samples, num_dimensions].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Xts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.
         - Yts : torch.Tensor (1D or 2D) or None
            Tensor of validation targets, of shape [num_test_samples, num_outputs].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Yts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.

        Returns:
        --------
         - model: Falkon
            The fitted model
        """
        if X.size(0) != Y.size(0):
            raise ValueError("X and Y must have the same number of "
                             "samples (found %d and %d)" %
                             (X.size(0), Y.size(0)))
        if Y.dim() == 1:
            Y = torch.unsqueeze(Y, 1)
        if Y.dim() != 2:
            raise ValueError("Y is expected 1D or 2D. Found %dD." % (Y.dim()))
        if not check_same_dtype(X, Y):
            raise TypeError("X and Y must have the same data-type.")

        dtype = X.dtype
        # Decide whether to use CUDA for preconditioning based on M
        _use_cuda_preconditioner = (
                self.use_cuda_ and
                (not self.cpu_preconditioner) and
                self.M >= get_min_cuda_preconditioner_size(dtype)
        )
        _use_cuda_mmv = (
                self.use_cuda_ and
                X.shape[0] * X.shape[1] * self.M / self.num_gpus >= get_min_cuda_mmv_size(dtype)
        )

        self.fit_times_ = []
        self.ny_points_ = None
        self.alpha_ = None

        t_s = time.time()
        ny_points = self.center_selection.select(X, None, self.M)
        if self.use_cuda_:
            ny_points = ny_points.pin_memory()

        with TicToc("Calcuating Preconditioner of size %d" % (self.M), debug=self.debug):
            precond_opt = self.extra_opt_.copy()
            precond_opt['use_cpu'] = not _use_cuda_preconditioner
            if self.extra_opt_.debug:
                print("Preconditioner will run on %s" %
                      ("CPU" if precond_opt['use_cpu'] else ("%d GPUs" % self.num_gpus)))
            precond = falkon.precond.FalkonPreconditioner(self.penalty, self.kernel, precond_opt)
            precond.init(ny_points)

        if _use_cuda_mmv:
            # Cache must be emptied to ensure enough memory is visible to the optimizer
            torch.cuda.empty_cache()
            X = X.pin_memory()

        # Decide whether it's worthwile to pre-compute the k_NM kernel.
        # If we precompute K_NM, each CG iteration costs
        # Given a single kernel evaluation between two D-dimensional vectors
        # costs D, at CG iteration we must perform N*M kernel evaluations.
        # Other than the kernel evaluations we must perform two matrix-vector
        # products 2(N*M*T) and a bunch of triangular solves.
        #
        # So if we precompute we have 2*(N*M*T), othewise we also have N*M*D
        # but precomputing costs us N*M memory.
        # So heuristic is the following:
        #  - If D is large (e.g. > 100) check if RAM is sufficient
        #  - If RAM is sufficient precompute
        #  - Otherwise do not precompute
        Knm = None
        if X.size(1) > 1200:
            necessary_ram = X.size(0) * ny_points.size(0) * sizeof_dtype(dtype)
            opt = self.extra_opt_.copy()
            opt['use_cpu'] = True
            cpu_info = get_device_info(opt)
            available_ram = min(
                self.extra_opt_.max_cpu_mem, cpu_info[-1].free_memory) * 0.9
            del opt

            if available_ram > necessary_ram:
                if self.debug:
                    print("%d*%d Kernel matrix will be stored" %
                          (X.size(0), ny_points.size(0)))
                Knm = self.kernel(X, ny_points, opt=self.extra_opt_)
                # TODO: Maybe we should do the same for Kts, but this complicates
                #       checks for fitting in memory
            elif self.debug:
                print(
                    "Cannot store full kernel matrix: not enough memory (have %.2fGB, need %.2fGB)" %
                    (available_ram / 2 ** 30, necessary_ram / 2 ** 30))
        self.fit_times_.append(time.time() - t_s)  # Preparation time

        # Here we define the callback function which will run at the end
        # of conjugate gradient iterations. This function computes and
        # displays the validation error.
        val_cback = None
        if self.error_fn is not None and self.error_every is not None:
            def val_cback(it, beta, train_time):
                self.fit_times_.append(self.fit_times_[0] + train_time)
                if it % self.error_every != 0:
                    print("Iteration %3d - Elapsed %.1fs" % (it, self.fit_times_[-1]), flush=True)
                    return
                err_str = "training" if Xts is None or Yts is None else "validation"
                alpha = precond.apply(beta)
                # Compute error: can be train or test;
                if Xts is not None and Yts is not None:
                    pred = self._predict(Xts, ny_points, alpha)
                    err = self.error_fn(Yts, pred)
                else:
                    pred = self._predict(X, ny_points, alpha)
                    err = self.error_fn(Y, pred)
                err_name = "error"
                if isinstance(err, tuple) and len(err) == 2:
                    err, err_name = err
                print("Iteration %3d - Elapsed %.1fs - %s %s: %.4f" %
                      (it, self.fit_times_[-1], err_str, err_name, err), flush=True)

        # Start with the falkon algorithm
        with TicToc('Computing Falkon iterations', debug=self.debug):
            optim_opt = self.extra_opt_.copy()
            optim_opt['use_cpu'] = not _use_cuda_mmv
            if self.extra_opt_.debug:
                print("Optimizer will run on %s" %
                      ("CPU" if optim_opt['use_cpu'] else ("%d GPUs" % self.num_gpus)), flush=True)
            optim = falkon.optim.FalkonConjugateGradient(self.kernel, precond, optim_opt)
            if Knm is not None:
                beta = optim.solve(
                    Knm, None, Y, self.penalty, initial_solution=None,
                    max_iter=self.maxiter, callback=val_cback)
            else:
                beta = optim.solve(
                    X, ny_points, Y, self.penalty, initial_solution=None,
                    max_iter=self.maxiter, callback=val_cback)

            self.alpha_ = precond.apply(beta)
            self.ny_points_ = ny_points
        return self

    def _predict(self, X, ny_points, alpha):
        if ny_points is None:
            # Then X is the kernel itself
            return X @ alpha
        _use_cuda_mmv = (
                self.use_cuda_ and
                X.shape[0] * X.shape[1] * self.M / self.num_gpus >= get_min_cuda_mmv_size(X.dtype)
        )
        mmv_opt = self.extra_opt_.copy()
        mmv_opt['use_cpu'] = not _use_cuda_mmv
        return self.kernel.mmv(X, ny_points, alpha, opt=mmv_opt)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the outputs on given test points.

        Parameters:
        -----------
         - X : torch.Tensor (2D)
            Tensor of test data points, of shape [num_samples, num_dimensions].
            If X is in Fortran order (i.e. column-contiguous) then we can avoid
            an extra copy of the data.

        Returns:
        --------
         - predictions : torch.Tensor (2D)
            Prediction tensor of shape [num_samples, num_outputs] for all
            data points.
        """
        if self.alpha_ is None or self.ny_points_ is None:
            raise RuntimeError(
                "Falkon has not been trained. `predict` must be called after `fit`.")

        return self._predict(X, self.ny_points_, self.alpha_)

    def __repr__(self, **kwargs):
        return super().__repr__(N_CHAR_MAX=5000)
