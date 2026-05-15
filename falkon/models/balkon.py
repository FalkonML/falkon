import dataclasses
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

import falkon
from falkon.models.model_utils import FalkonBase
from falkon.options import FalkonOptions
from falkon.preconditioner.blk_pc import BalkonPreconditioner
from falkon.utils import TicToc
from falkon.utils.devices import get_device_info

__all__ = ("Balkon",)


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


class Balkon(FalkonBase):
    def __init__(
        self,
        kernel: falkon.kernels.Kernel,
        penalty: float,
        M: int,
        block_size: int,
        center_selection: str | falkon.center_selection.CenterSelector = "uniform",
        maxiter: int = 20,
        seed: int | None = None,
        error_fn: Callable[[torch.Tensor, torch.Tensor], Any | tuple[Any, str]] | None = None,
        error_every: int | None = 1,
        options: FalkonOptions | None = None,
    ):
        super().__init__(kernel, M, center_selection, seed, error_fn, error_every, options)
        self.penalty = penalty
        self.maxiter = maxiter
        if M < block_size:
            raise ValueError(
                f"Preconditioner block size must be smaller or "
                f"equal to the number of centers. Found {M} centers "
                f"and block size {block_size}."
            )
        self.block_size = block_size
        self._init_cuda()
        self.precond: BalkonPreconditioner | None = None

    def _reset_state(self):
        super()._reset_state()
        self.precond = None

    def init_pc(
        self,
        ny_points: Tensor,
        n: int,
        use_cuda_pc: bool,
    ) -> BalkonPreconditioner:
        num_centers = ny_points.shape[0]
        with TicToc(f"Calcuating Preconditioner of size {num_centers}", debug=self.options.debug):
            pc_opt: FalkonOptions = dataclasses.replace(self.options, use_cpu=not use_cuda_pc)
            if pc_opt.debug:
                dev_str = "CPU" if pc_opt.use_cpu else f"{self.num_gpus} GPUs"
                print(f"Preconditioner will run on {dev_str}")
            pc = BalkonPreconditioner(self.penalty, self.kernel, data_size=n, block_size=self.block_size, opt=pc_opt)
            pc.init(ny_points)
        return pc

    def init_kernel_matrix(self, X: Tensor, ny_pts: Tensor) -> falkon.kernels.Kernel:
        """
        Decide whether to store the full kernel. If dimensions are such that it is convenient
        to precompute it, it is saved in a :class:`PrecomputedKernel` which is used for
        subsequent computations. Otherwise return the original kernel..
        """
        k_opt = dataclasses.replace(self.options, use_cpu=True)
        cpu_info = get_device_info(k_opt)
        available_ram = min(k_opt.max_cpu_mem, cpu_info[-1].free_memory) * 0.9
        kernel = self.kernel
        if self._can_store_knm(X, ny_pts, available_ram):
            Knm = self.kernel(X, ny_pts, opt=self.options)
            kernel = falkon.kernels.PrecomputedKernel(Knm, opt=self.options)
        return kernel

    def run_solver(
        self,
        use_cuda: bool,
        kernel: falkon.kernels.Kernel,
        X: Tensor,
        Y: Tensor,
        ny_pts: Tensor,
        warm_start: Tensor | None,
        cb: Callable,
    ) -> Tensor:
        assert self.precond is not None
        with TicToc("Computing Falkon iterations", debug=self.options.debug):
            o_opt: FalkonOptions = dataclasses.replace(self.options, use_cpu=not use_cuda)
            if o_opt.debug:
                optim_dev_str = "CPU" if o_opt.use_cpu else f"{self.num_gpus} GPUs"
                print(f"Optimizer will run on {optim_dev_str}", flush=True)
            optim = falkon.optim.BalkonConjugateGradient(kernel, self.precond, o_opt)
            alpha = optim.solve(
                X, ny_pts, Y, self.penalty, initial_solution=warm_start, max_iter=self.maxiter, callback=cb
            )

        return alpha

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Xts: torch.Tensor | None = None,
        Yts: torch.Tensor | None = None,
        warm_start: torch.Tensor | None = None,
    ):
        X, Y, Xts, Yts = self._check_fit_inputs(X, Y, Xts, Yts)
        self._reset_state()

        # Start training timer
        t_s = time.time()

        with torch.autograd.inference_mode():
            ny_points, ny_indices = self.center_selection.select_indices(X, None)
            num_centers = ny_points.shape[0]
            pc_block_size = num_centers // self.block_size

            # Decide whether to use CUDA for preconditioning and iterations
            _use_cuda_preconditioner = (
                self.use_cuda_
                and (not self.options.cpu_preconditioner)
                and pc_block_size >= get_min_cuda_preconditioner_size(X.dtype, self.options)
            )
            tot_mmv_mem_usage = X.shape[0] * X.shape[1] * num_centers  # N*D*M
            _use_cuda_mmv = self.use_cuda_ and tot_mmv_mem_usage / self.num_gpus >= get_min_cuda_mmv_size(
                X.dtype, self.options
            )

            if self.use_cuda_:
                ny_points = ny_points.pin_memory()

            self.precond = self.init_pc(ny_points, n=X.shape[0], use_cuda_pc=_use_cuda_preconditioner)

            if _use_cuda_mmv:
                # Cache must be emptied to ensure enough memory is visible to the optimizer
                torch.cuda.empty_cache()
                X = X.pin_memory()

            calc_kernel = self.init_kernel_matrix(X, ny_points)
            self.fit_times_.append(time.time() - t_s)  # Preparation time

            # Define the callback function which runs after each CG iteration. Optionally computes
            # and displays the validation error.
            validation_cback = None
            if self.error_fn is not None and self.error_every is not None:
                validation_cback = self._get_callback_fn(X, Y, Xts, Yts, ny_points, self.precond)

            alpha = self.run_solver(_use_cuda_mmv, calc_kernel, X, Y, ny_points, warm_start, validation_cback)
            self.alpha_, self.ny_points_ = alpha, ny_points
        return self

    def _predict(self, X, ny_points, alpha: torch.Tensor) -> torch.Tensor:
        with torch.autograd.inference_mode():
            num_centers = alpha.shape[0]
            tot_mmv_mem_usage = X.shape[0] * X.shape[1] * num_centers
            _use_cuda_mmv = alpha.device.type == "cuda" or (
                self.use_cuda_ and tot_mmv_mem_usage / self.num_gpus >= get_min_cuda_mmv_size(X.dtype, self.options)
            )
            mmv_opt = dataclasses.replace(self.options, use_cpu=not _use_cuda_mmv)
            return self.kernel.mmv(X, ny_points, alpha, opt=mmv_opt)

    def _params_to_original_space(self, params, preconditioner):
        return params

    def to(self, device):
        self.alpha_ = self.alpha_.to(device)
        self.ny_points_ = self.ny_points_.to(device)
        return self
