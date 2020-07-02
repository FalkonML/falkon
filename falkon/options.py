from dataclasses import dataclass

import numpy as np
import torch

__all__ = ("BaseOptions", "KeopsOptions", "ConjugateGradientOptions", "PreconditionerOptions",
           "LauumOptions", "CholeskyOptions", "FalkonOptions")

_docs = {
    "base":
    """
debug (default False)
    When set to `True`, the estimators will print extensive debugging information.
    Set it if you want to dig deeper.
use_cpu (default False)
    When set to `True` forces Falkon not to use the GPU. If this option is not set,
    and no GPU is available, Falkon will issue a warning.
max_gpu_mem
    The maximum GPU memory (in bytes) that Falkon may use. If not set, Falkon will
    use all available memory.
max_cpu_mem
    The maximum CPU RAM (in bytes) that Falkon may use. If not set, Falkon will
    use all available memory. This option is not a strict bound (due to the nature
    of memory management in Python).
compute_arch_speed (default False)
    When running Falkon on a machine with multiple GPUs which have a range of different
    performance characteristics, setting this option to `True` may help subdivide the
    workload better: the performance of each accelerator will be evaluated on startup,
    then the faster devices will receive more work than the slower ones.
    If this is not the case, do not set this option since evaluating accelerator performance
    increases startup times.
no_single_kernel (default True)
    Whether the kernel should always be evaluated in double precision.
    If set to `False`, kernel evaluations will be faster but less precise (note that this
    referes only to calculations involving the full kernel matrix, not to kernel-vector
    products).
min_cuda_pc_size_32 (default 10000)
    If M (the number of Nystroem centers) is lower than `min_cuda_pc_size_32`, falkon will
    run the preconditioner on the CPU. Otherwise, if CUDA is available, falkon will try 
    to run the preconditioner on the GPU. This setting is valid for data in single 
    (float32) precision.
    Along with the `min_cuda_iter_size_32` setting, this determines a cutoff for running
    Falkon on the CPU or the GPU. Such cutoff is useful since for small-data problems
    running on the CPU may be faster than running on the GPU. If your data is close to the
    cutoff, it may be worth experimenting with running on the CPU and on the GPU to check
    which side is faster. This will depend on the exact hardware.
min_cuda_pc_size_64 (default 30000)
    If M (the number of Nystroem centers) is lower than `min_cuda_pc_size_64`, falkon will
    run the preconditioner on the CPU. Otherwise, if CUDA is available, falkon will try 
    to run the preconditioner on the GPU. This setting is valid for data in double 
    (float64) precision.
    Along with the `min_cuda_iter_size_64` setting, this determines a cutoff for running
    Falkon on the CPU or the GPU. Such cutoff is useful since for small-data problems
    running on the CPU may be faster than running on the GPU. If your data is close to the
    cutoff, it may be worth experimenting with running on the CPU and on the GPU to check
    which side is faster. This will depend on the exact hardware.
min_cuda_iter_size_32 (default 300_000_000)
    If the data size (measured as the product of M, and the dimensions of X) is lower than
    `min_cuda_iter_size_32`, falkon will run the conjugate gradient iterations on the CPU.
    For example, with the default setting, the CPU-GPU threshold is set at a dataset
    with 10k points, 10 dimensions, and 3k Nystroem centers. A larger dataset, or the use
    of more centers, will cause the conjugate gradient iterations to run on the GPU.
    This setting is valid for data in single (float32) precision.
min_cuda_iter_size_64 (default 900_000_000)
    If the data size (measured as the product of M, and the dimensions of X) is lower than
    `min_cuda_iter_size_64`, falkon will run the conjugate gradient iterations on the CPU.
    For example, with the default setting, the CPU-GPU threshold is set at a dataset
    with 30k points, 10 dimensions, and 3k Nystroem centers. A larger dataset, or the use
    of more centers, will cause the conjugate gradient iterations to run on the GPU.
    This setting is valid for data in double (float64) precision.
    """,
    "keops":
    """
keops_acc_dtype (default "auto")
    A string describing the accumulator data-type for KeOps.
    For more information refer to the
    `KeOps documentation <https://www.kernel-operations.io/keops/python/api/pytorch/Genred_torch.html?highlight=genred#pykeops.torch.Genred>`_
keops_sum_scheme (default "auto")
    Accumulation scheme for KeOps. For more information refer to the
    `KeOps documentation <https://www.kernel-operations.io/keops/python/api/pytorch/Genred_torch.html?highlight=genred#pykeops.torch.Genred>`_
keops_active : (default "auto")
    Whether to use or not to use KeOps. Three settings are allowed, specified by strings:
    'auto' (the default setting) means that KeOps will be used if it is installed correctly,
    'no' means keops will not be used, nor will it be imported, and 'force' means that if KeOps is
    not installed an error will be raised.
    """,
    "cg":
    """
cg_epsilon_32 (default 1e-7)
    Small added epsilon to prevent divide-by-zero errors in the conjugate gradient algorithm. 
    Used for single precision data-types
cg_epsilon_64 (default 1e-15)
    Small added epsilon to prevent divide-by-zero errors in the conjugate gradient algorithm. 
    Used for double precision data-types
cg_tolerance
    Maximum change in model parameters between iterations. If less change than `cg_tolerance`
    is detected, then we regard the optimization as converged.
cg_full_gradient_every
    How often to calculate the full gradient in the conjugate gradient algorithm. Full-gradient
    iterations take roughly twice the time as normal iterations, but they reset the error
    introduced by the other iterations.
    """,
    "pc":
    """
pc_epsilon_32
    Epsilon used to increase the diagonal dominance of a matrix before its
    Cholesky decomposition (for single-precision data types).
pc_epsilon_64
    Epsilon used to increase the diagonal dominance of a matrix before its
    Cholesky decomposition (for double-precision data types).
cpu_preconditioner
    Whether the preconditioner should be computed on the CPU. This setting overrides
    the :attr:`FalkonOptions.use_cpu` option.
    """,
    "lauum":
    """
lauum_par_blk_multiplier
    Minimum number of tiles per-GPU for the LAUUM algorithm. This can be set quite high (e.g. 8)
    too much performance degradation. Optimal settings will depend on the number of GPUs.
    """,
    "chol":
    """
chol_force_in_core
    Whether to force in-core execution of the Cholesky decomposition. This will
    not work with matrices bigger than GPU memory.
chol_force_ooc
    Whether to force out-of-core (parallel) execution for the POTRF algorithm, 
    even on matrices which fit in-GPU-core.
chol_par_blk_multiplier
    Minimum number of tiles per-GPU in the out-of-core, GPU-parallel POTRF algorithm.
    """
}


@dataclass
class BaseOptions():
    """A set of options which are common to different modules
    """
    debug: bool = False
    use_cpu: bool = False
    max_gpu_mem: float = np.inf
    max_cpu_mem: float = np.inf
    compute_arch_speed: bool = False
    no_single_kernel: bool = True
    min_cuda_pc_size_32: int = 10000
    min_cuda_pc_size_64: int = 30000
    min_cuda_iter_size_32: int = 10_000 * 10 * 3_000
    min_cuda_iter_size_64: int = 30_000 * 10 * 3_000

    def get_base_options(self):
        return BaseOptions(debug=self.debug,
                           use_cpu=self.use_cpu,
                           max_gpu_mem=self.max_gpu_mem,
                           max_cpu_mem=self.max_cpu_mem,
                           no_single_kernel=self.no_single_kernel,
                           compute_arch_speed=self.compute_arch_speed,
                           min_cuda_pc_size_32=self.min_cuda_pc_size_32,
                           min_cuda_pc_size_64=self.min_cuda_pc_size_64,
                           min_cuda_iter_size_32=self.min_cuda_iter_size_32,
                           min_cuda_iter_size_64=self.min_cuda_iter_size_64)


@dataclass
class KeopsOptions():
    """A set of options which relate to usage of KeOps
    """
    keops_acc_dtype: str = "auto"
    keops_sum_scheme: str = "auto"
    keops_active: str = "auto"

    def get_keops_options(self):
        return KeopsOptions(keops_acc_dtype=self.keops_acc_dtype,
                            keops_sum_scheme=self.keops_sum_scheme,
                            keops_active=self.keops_active)


@dataclass
class ConjugateGradientOptions():
    """A set of options related to conjugate gradient optimization
    """
    cg_epsilon_32: float = 1e-7
    cg_epsilon_64: float = 1e-15
    cg_tolerance: float = 1e-7
    cg_full_gradient_every: int = 10

    def cg_epsilon(self, dtype):
        if dtype == torch.float32:
            return self.cg_epsilon_32
        elif dtype == torch.float64:
            return self.cg_epsilon_64
        else:
            raise TypeError("Data-type %s invalid" % (dtype))

    def get_conjgrad_options(self):
        return ConjugateGradientOptions(cg_epsilon_32=self.cg_epsilon_32,
                                        cg_epsilon_64=self.cg_epsilon_64,
                                        cg_tolerance=self.cg_tolerance,
                                        cg_full_gradient_every=self.cg_full_gradient_every)


@dataclass
class PreconditionerOptions():
    """Options related to calculation of the preconditioner

    See Also
    --------
    :class:`falkon.preconditioner.FalkonPreconditioner` :
        Preconditioner class which uses these options
    """
    pc_epsilon_32: float = 1e-5
    pc_epsilon_64: float = 1e-13
    cpu_preconditioner: bool = False

    def pc_epsilon(self, dtype):
        if dtype == torch.float32:
            return self.pc_epsilon_32
        elif dtype == torch.float64:
            return self.pc_epsilon_64
        else:
            raise TypeError("Data-type %s invalid" % (dtype))

    def get_pc_options(self):
        return PreconditionerOptions(pc_epsilon_32=self.pc_epsilon_32,
                                     pc_epsilon_64=self.pc_epsilon_64,
                                     cpu_preconditioner=self.cpu_preconditioner)


@dataclass(frozen=False)
class LauumOptions():
    """Options related to the out-of-core LAUUM (triangular matrix multiplication) operation

    """
    lauum_par_blk_multiplier: int = 8

    def get_lauum_options(self):
        return LauumOptions(lauum_par_blk_multiplier=self.lauum_par_blk_multiplier)


@dataclass(frozen=False)
class CholeskyOptions():
    """Options related to the out-of-core POTRF (Cholesky decomposition) operation

    """
    chol_force_in_core: bool = False
    chol_force_ooc: bool = False
    chol_par_blk_multiplier: int = 2

    def get_chol_options(self):
        return CholeskyOptions(chol_force_in_core=self.chol_force_in_core,
                               chol_force_ooc=self.chol_force_ooc,
                               chol_par_blk_multiplier=self.chol_par_blk_multiplier)


@dataclass()
class FalkonOptions(BaseOptions, ConjugateGradientOptions, PreconditionerOptions, LauumOptions,
                    CholeskyOptions, KeopsOptions):
    """Global options for Falkon."""
    pass


# Fix documentation: FalkonOptions must inherit all params from its super-classes.
def _reset_doc(cls, params):
    cls.__doc__ = "%s\n\nParameters\n----------%s\n" % (cls.__doc__, params)


_reset_doc(BaseOptions, _docs["base"])
_reset_doc(KeopsOptions, _docs["keops"])
_reset_doc(ConjugateGradientOptions, _docs["cg"])
_reset_doc(PreconditionerOptions, _docs["pc"])
_reset_doc(LauumOptions, _docs["lauum"])
_reset_doc(CholeskyOptions, _docs["chol"])


FalkonOptions.__doc__ = "%s\n\nParameters\n----------%s%s%s%s%s%s\n" % (
    FalkonOptions.__doc__, _docs["base"], _docs["keops"], _docs["cg"], _docs["pc"], _docs["lauum"],
    _docs["chol"])
