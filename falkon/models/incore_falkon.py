import time
import warnings
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

import falkon
from falkon import FalkonOptions
from falkon.kernels import Kernel
from falkon.kernels.precomputed_kernel import PrecomputedKernel
from falkon.preconditioner import FalkonPreconditioner
from falkon.sparse import SparseTensor
from falkon.utils.devices import get_device_info
from falkon.utils.helpers import check_same_device

from .falkon import Falkon

__all__ = ("InCoreFalkon",)


class InCoreFalkon(Falkon):
    """In GPU core Falkon Kernel Ridge Regression solver.

    This estimator object solves approximate kernel ridge regression problems with Nystroem
    projections and a fast optimization algorithm as described in :ref:`[1] <flk_1>`, :ref:`[2] <flk_2>`.

    Multiclass and multiple regression problems can all be tackled
    with this same object, for example by encoding multiple classes
    in a one-hot target matrix.

    Compared to the base `falkon.models.Falkon` estimator, the `InCoreFalkon` estimator
    is designed to work fully within the GPU, performing no data-copies between CPU and GPU. As
    such, it is more constraining than the base estimator, but **has better performance on smaller
    problems**.
    In particular, the constraints are that:

     - the input data must be on a single GPU, when calling `InCoreFalkon.fit`;
     - the data, preconditioner, kernels, etc. must all fit on the same GPU.

    Using multiple GPUs is not possible with this model.

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
        computational resources. You can either specify the number of centers
        by setting this parameter, or by passing to this constructor a `CenterSelctor` class
        instance.
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
    weight_fn : Callable or None
        A function for giving different weights to different samples. This is used
        for weighted least-squares, it should accept three arguments: `Y`, `X`, `indices` which
        represent the samples for which weights need to be computed, and return a vector of
        weights corresponding to the input targets.

        As an example, in the setting of binary classification Y can be -1 or +1. To give more
        importance to errors on the negative class, pass a `weight_fn` which returns 2 whenever
        the target is -1.
    options : FalkonOptions
        Additional options used by the components of the Falkon solver. Individual options
        are documented in :mod:`falkon.options`.

    Examples
    --------
    Running :class:`InCoreFalkon` on a randomly generated dataset

    >>> X = torch.randn(1000, 10).cuda()
    >>> Y = torch.randn(1000, 1).cuda()
    >>> kernel = falkon.kernels.GaussianKernel(3.0)
    >>> options = FalkonOptions(use_cpu=True)
    >>> model = InCoreFalkon(kernel=kernel, penalty=1e-6, M=500, options=options)
    >>> model.fit(X, Y)
    >>> preds = model.predict(X)
    >>> assert preds.is_cuda

    References
    ----------
     - Alessandro Rudi, Luigi Carratino, Lorenzo Rosasco, "FALKON: An optimal large
       scale kernel method," Advances in Neural Information Processing Systems 29, 2017.
     - Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi,
       "Kernel methods through the roof: handling billions of points efficiently,"
       Advancs in Neural Information Processing Systems, 2020.
    """

    def __init__(
        self,
        kernel: falkon.kernels.Kernel,
        penalty: float,
        M: int,
        center_selection: Union[str, falkon.center_selection.CenterSelector] = "uniform",
        maxiter: int = 20,
        seed: Optional[int] = None,
        error_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Union[float, Tuple[float, str]]]] = None,
        error_every: Optional[int] = 1,
        weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        options: Optional[FalkonOptions] = None,
    ):
        super().__init__(kernel, penalty, M, center_selection, maxiter, seed, error_fn, error_every, weight_fn, options)
        if not self.use_cuda_:
            raise RuntimeError(
                "Cannot instantiate InCoreFalkon when CUDA is not available. "
                "If CUDA is present on your system, make sure to set "
                "'use_cpu=False' in the `FalkonOptions` object."
            )

    def _reset_state(self):
        super()._reset_state()
        self.beta_ = None
        self.precond = None

    def _check_fit_inputs(self, X, Y, Xts, Yts):
        if not check_same_device(X, Y, Xts, Yts) or (not X.is_cuda):
            raise ValueError("All tensors for fitting InCoreFalkon must be CUDA tensors, located on the same GPU.")
        return super()._check_fit_inputs(X, Y, Xts, Yts)

    def _check_predict_inputs(self, X):
        if not check_same_device(X, self.alpha_):
            raise ValueError(f"X must be on device {self.alpha_.device}")
        return super()._check_predict_inputs(X)

    def init_pc(
        self,
        ny_points: Union[Tensor, SparseTensor],
        use_cuda_pc: bool,
        X: Tensor,
        Y: Tensor,
        ny_indices: Optional[Tensor] = None,
    ) -> FalkonPreconditioner:
        assert use_cuda_pc is True
        pc_stream = torch.cuda.Stream(X.device)
        with torch.cuda.stream(pc_stream):
            pc = super().init_pc(ny_points=ny_points, X=X, Y=Y, use_cuda_pc=True, ny_indices=ny_indices)
        pc_stream.synchronize()
        return pc

    def init_kernel_matrix(self, X: Tensor, ny_pts: Tensor) -> Kernel:
        """
        Decide whether to store the full kernel. If dimensions are such that it is convenient
        to precompute it, it is saved in a :class:`PrecomputedKernel` which is used for
        subsequent computations. Otherwise return the original kernel..
        """
        gpu_info = get_device_info(self.options)[X.device.index]
        available_ram = min(self.options.max_gpu_mem, gpu_info.free_memory) * 0.9
        kernel = self.kernel
        if self._can_store_knm(X, ny_pts, available_ram):
            Knm = self.kernel(X, ny_pts, opt=self.options)
            kernel = PrecomputedKernel(Knm, opt=self.options)
        return kernel

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Xts: Optional[torch.Tensor] = None,
        Yts: Optional[torch.Tensor] = None,
        warm_start: Optional[torch.Tensor] = None,
    ):
        """Fits the Falkon KRR model.

        Parameters
        -----------
        X : torch.Tensor
            The tensor of training data, of shape [num_samples, num_dimensions].
            If X is in Fortran order (i.e. column-contiguous) then we can avoid
            an extra copy of the data. Must be a CUDA tensor.
        Y : torch.Tensor
            The tensor of training targets, of shape [num_samples, num_outputs].
            If X and Y represent a classification problem, Y can be encoded as a one-hot
            vector.
            If Y is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data. Must be a CUDA tensor.
        Xts : torch.Tensor or None
            Tensor of validation data, of shape [num_test_samples, num_dimensions].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Xts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data. Must be a CUDA tensor.
        Yts : torch.Tensor or None
            Tensor of validation targets, of shape [num_test_samples, num_outputs].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Yts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data. Must be a CUDA tensor.
        warm_start : torch.Tensor or None
            Specify a starting point for the conjugate gradient optimizer. If not specified, the
            initial point will be a tensor filled with zeros.
            Be aware that the starting point should not be in the parameter space, but in the
            preconditioner space (i.e. if initializing from a previous Falkon object, use the
            `beta_` field, not `alpha_`).

        Returns
        --------
        model: InCoreFalkon
            The fitted model

        """
        # Fixes a synchronization bug which occurs when re-using center selector.
        torch.cuda.synchronize()
        X, Y, Xts, Yts = self._check_fit_inputs(X, Y, Xts, Yts)
        self._reset_state()

        # Start training timer
        t_s = time.time()

        with torch.autograd.inference_mode():
            if self.weight_fn is None:  # don't need indices.
                ny_points, ny_indices = self.center_selection.select(X, None), None
            else:
                ny_points, ny_indices = self.center_selection.select_indices(X, None)

            self.precond = self.init_pc(ny_points, True, X, Y, ny_indices)

            # Cache must be emptied to ensure enough memory is visible to the optimizer
            torch.cuda.empty_cache()

            calc_kernel = self.init_kernel_matrix(X, ny_points)
            self.fit_times_.append(time.time() - t_s)  # Preparation time

            # Define the callback function which runs after each CG iteration. Optionally computes
            # and displays the validation error.
            validation_cback = None
            if self.error_fn is not None and self.error_every is not None:
                validation_cback = self._get_callback_fn(X, Y, Xts, Yts, ny_points, self.precond)

            alpha, beta = self.run_solver(True, calc_kernel, X, Y, ny_points, warm_start, validation_cback)
            self.alpha_, self.beta_, self.ny_points_ = alpha, beta, ny_points
        return self

    def _predict(self, X, ny_points, alpha):
        with torch.autograd.inference_mode():
            if ny_points is None:
                warnings.warn("This code-path is deprecated and may be removed. Nys_points must be specified.")
                # Then X is the kernel itself
                return X @ alpha
            return self.kernel.mmv(X, ny_points, alpha, opt=self.options)

    def _params_to_original_space(self, params, preconditioner):
        return preconditioner.apply(params)
