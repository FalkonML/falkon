import dataclasses
import time
from typing import Union, Optional, Callable

import torch

import falkon
from falkon.models.model_utils import FalkonBase
from falkon.options import *
from falkon.utils import TicToc
from falkon.utils.devices import get_device_info

__all__ = ("Falkon",)


def get_min_cuda_preconditioner_size(dt, opt: FalkonOptions) -> int:
    if dt == torch.float32:
        return opt.min_cuda_pc_size_32
    else:
        return opt.min_cuda_pc_size_64


def get_min_cuda_mmv_size(dt, opt: FalkonOptions) -> int:
    if dt == torch.float32:
        return opt.min_cuda_iter_size_32
    else:
        return opt.min_cuda_iter_size_64


class Falkon(FalkonBase):
    """Falkon Kernel Ridge Regression solver.

    This estimator object solves approximate kernel ridge regression problems with Nystroem
    projections and a fast optimization algorithm as described in [flk_1]_, [flk_2]_.

    Multiclass and multiple regression problems can all be tackled
    with this same object, for example by encoding multiple classes
    in a one-hot target matrix.

    Parameters
    -----------
    kernel
        Object representing the kernel function used for KRR.
    penalty : float
        Amount of regularization to apply to the problem.
        This parameter must be greater than 0.
    M : int
        The number of Nystrom centers to pick. `M` must be positive,
        and lower than the total number of training points. A larger
        `M` will typically lead to better accuracy but will use more
        computational resources.
    center_selection : str or falkon.center_selection.CenterSelector
        The center selection algorithm. Implemented is only 'uniform'
        selection which can choose each training sample with the same
        probability.
    maxiter : int
        The number of iterations to run the optimization for. Usually
        fewer than 20 iterations are necessary, however this is problem
        dependent.
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
    weight_fun : Callable or None
        A function with one argument (a torch.Tensor containing the targets (or a subset of it)) which returns a vector of weights.
        If set to None, it will never be used.  
    options : FalkonOptions
        Additional options used by the components of the Falkon solver. Individual options
        are documented in :mod:`falkon.options`.

    Examples
    --------
    Running Falkon on a random dataset

    >>> X = torch.randn(1000, 10)
    >>> Y = torch.randn(1000, 1)
    >>> kernel = falkon.kernels.GaussianKernel(3.0)
    >>> options = FalkonOptions(use_cpu=True)
    >>> model = Falkon(kernel=kernel, penalty=1e-6, M=500, options=options)
    >>> model.fit(X, Y)
    >>> preds = model.predict(X)

    References
    ----------
    .. [flk_1] Alessandro Rudi, Luigi Carratino, Lorenzo Rosasco, "FALKON: An optimal large
       scale kernel method," Advances in Neural Information Processing Systems 29, 2017.
    .. [flk_2] Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi,
       "Kernel methods through the roof: handling billions of points efficiently,"
       arXiv:2006.10350, 2020.

    """

    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty: float,
                 M: int,
                 center_selection: Union[str, falkon.center_selection.CenterSelector] = 'uniform',
                 maxiter: int = 20,
                 seed: Optional[int] = None,
                 error_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
                 error_every: Optional[int] = 1,
                 weight_fun = None,
                 options: Optional[FalkonOptions] = None,
                 ):
        super().__init__(kernel, M, center_selection, seed, error_fn, error_every, options)
        self.penalty = penalty
        self.maxiter = maxiter
        self.weight_fun = weight_fun
        self._init_cuda()

    def fit(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            Xts: Optional[torch.Tensor] = None,
            Yts: Optional[torch.Tensor] = None):
        """Fits the Falkon KRR model.

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
        model: Falkon
            The fitted model
        """
        X, Y, Xts, Yts = self._check_fit_inputs(X, Y, Xts, Yts)

        dtype = X.dtype

        # Decide whether to use CUDA for preconditioning based on M
        _use_cuda_preconditioner = (
            self.use_cuda_ and
            (not self.options.cpu_preconditioner) and
            self.M >= get_min_cuda_preconditioner_size(dtype, self.options)
        )
        _use_cuda_mmv = (
            self.use_cuda_ and
            X.shape[0] * X.shape[1] * self.M / self.num_gpus >= get_min_cuda_mmv_size(dtype, self.options)
        )

        self.fit_times_ = []
        self.ny_points_ = None
        self.alpha_ = None

        t_s = time.time()
        # noinspection PyTypeChecker
        if self.weight_fun is None:
            ny_points: Union[torch.Tensor, falkon.sparse.SparseTensor] = self.center_selection.select(X, None, self.M)
        else:
            ny_points, ny_indices = self.center_selection.select(X, None, self.M, True)
        
        if self.use_cuda_:
            ny_points = ny_points.pin_memory()

        with TicToc("Calcuating Preconditioner of size %d" % (self.M), debug=self.options.debug):
            pc_opt: FalkonOptions = dataclasses.replace(self.options,
                                                        use_cpu=not _use_cuda_preconditioner)
            if pc_opt.debug:
                print("Preconditioner will run on %s" %
                      ("CPU" if pc_opt.use_cpu else ("%d GPUs" % self.num_gpus)))
            if self.weight_fun is None:
                precond = falkon.preconditioner.FalkonPreconditioner(self.penalty, self.kernel, pc_opt, None)
            else:
                ny_weight_vec = self.weight_fun(Y[ny_indices])
                precond = falkon.preconditioner.FalkonPreconditioner(self.penalty, self.kernel, pc_opt,ny_weight_vec)
            precond.init(ny_points)

        if _use_cuda_mmv:
            # Cache must be emptied to ensure enough memory is visible to the optimizer
            torch.cuda.empty_cache()
            X = X.pin_memory()

        # K_NM storage decision
        k_opt = dataclasses.replace(self.options, use_cpu=True)
        cpu_info = get_device_info(k_opt)
        available_ram = min(k_opt.max_cpu_mem, cpu_info[-1].free_memory) * 0.9
        if self._can_store_knm(X, ny_points, available_ram):
            Knm = self.kernel(X, ny_points, opt=self.options)
        else:
            Knm = None
        self.fit_times_.append(time.time() - t_s)  # Preparation time

        # Here we define the callback function which will run at the end
        # of conjugate gradient iterations. This function computes and
        # displays the validation error.
        validation_cback = None
        if self.error_fn is not None and self.error_every is not None:
            validation_cback = self._get_callback_fn(X, Y, Xts, Yts, ny_points, precond)

        # Start with the falkon algorithm
        with TicToc('Computing Falkon iterations', debug=self.options.debug):
            o_opt: FalkonOptions = dataclasses.replace(self.options, use_cpu=not _use_cuda_mmv)
            if o_opt.debug:
                print("Optimizer will run on %s" %
                      ("CPU" if o_opt.use_cpu else ("%d GPUs" % self.num_gpus)), flush=True)
            if self.weight_fun is None:
                optim = falkon.optim.FalkonConjugateGradient(self.kernel, precond, o_opt)
            else:
                optim = falkon.optim.WFalkonConjugateGradient(self.kernel, precond, o_opt, self.weight_fun)
            
            if Knm is not None:
                beta = optim.solve(
                    Knm, None, Y, self.penalty, initial_solution=None,
                    max_iter=self.maxiter, callback=validation_cback)
            else:
                beta = optim.solve(
                    X, ny_points, Y, self.penalty, initial_solution=None,
                    max_iter=self.maxiter, callback=validation_cback)

            self.alpha_ = precond.apply(beta)
            self.ny_points_ = ny_points
        return self

    def _predict(self, X, ny_points, alpha: torch.Tensor) -> torch.Tensor:
        if ny_points is None:
            # Then X is the kernel itself
            return X @ alpha
        _use_cuda_mmv = (
            alpha.device.type == "cuda" or (
                self.use_cuda_ and
                X.shape[0] * X.shape[1] * self.M / self.num_gpus >= get_min_cuda_mmv_size(
                    X.dtype, self.options)
            )
        )
        mmv_opt = dataclasses.replace(self.options, use_cpu=not _use_cuda_mmv)
        return self.kernel.mmv(X, ny_points, alpha, opt=mmv_opt)

    def to(self, device):
        self.alpha_ = self.alpha_.to(device)
        self.ny_points_ = self.ny_points_.to(device)
        return self
