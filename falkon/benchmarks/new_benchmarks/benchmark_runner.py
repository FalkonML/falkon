import argparse
import datetime
import functools
import sys

import numpy as np
import torch

from falkon.benchmarks.common.benchmark_utils import Dataset, DataType
from falkon.benchmarks.common.datasets import get_cv_fn, get_load_fn
from falkon.benchmarks.common.error_metrics import get_err_fns

RANDOM_SEED = 123
EIGENPRO_BASE_PATH = "/home/giacomo/EigenPro"


def test_model(model, model_name, Xts, Yts, Xtr, Ytr, err_fns):
    test_preds = model.predict(Xts)
    train_preds = None
    if Xtr is not None:
        train_preds = model.predict(Xtr)
    test_errs, train_errs = [], []
    for err_fn in err_fns:
        test_err, test_err_name = err_fn(Yts, test_preds)
        test_errs.append(test_err)
        print(f"Test {model_name} {test_err_name}: {test_err:9.6f}", flush=True)
        if Xtr is not None and Ytr is not None:
            assert train_preds is not None
            train_err, train_err_name = err_fn(Ytr, train_preds)
            print(f"Train {model_name} {train_err_name}: {train_err:9.6f}", flush=True)
            train_errs.append(train_err)
    return test_errs, train_errs


def print_kfold_error_report(k, test_errs, train_errs, err_fns):
    print(f"Full errors: Test {test_errs} - Train {train_errs}")
    print()
    print(f"{k}-Fold Error Report")
    for err_fn_i in range(len(err_fns)):
        print(
            f"Final test errors: "
            f"{np.mean([e[err_fn_i] for e in test_errs]):.4f} +- "
            f"{np.std([e[err_fn_i] for e in test_errs]):4f}"
        )
        print(
            f"Final train errors: "
            f"{np.mean([e[err_fn_i] for e in train_errs]):.4f} +- "
            f"{np.std([e[err_fn_i] for e in train_errs]):.4f}"
        )
        print()


################
### EIGENPRO ###
def run_eigenpro(
    dset: Dataset,
    data_path: str,
    dtype: DataType | None,
    num_iter: int,
    num_centers: int,
    num_pc_centers: int,
    num_eigenvalues: int,
    kernel_sigma: float,
    kernel: str,
    kfold: int,
    seed: int,
):
    from falkon.utils import TicToc

    sys.path.append(EIGENPRO_BASE_PATH)
    import eigenpro.kernels as kernels # pyright: ignore[reportMissingImports]
    import eigenpro.utils.device as dev # pyright: ignore[reportMissingImports]
    from falkon.benchmarks.new_benchmarks.eigenpro_wrapper import EigenProWrapper

    torch.manual_seed(seed)
    np.random.seed(seed)

    if dtype is None:
        dtype = DataType.float32
    if kernel == "laplacian":
        kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=kernel_sigma)
    elif kernel == "gaussian":
        kernel_fn = lambda x, z: kernels.gaussian(x, z, bandwidth=kernel_sigma)
    else:
        raise ValueError(kernel)
    device = dev.Device.create(use_gpu_if_available=True)
    model = EigenProWrapper(device, dtype.to_torch_dtype(), kernel_fn, num_centers=num_centers,
                            num_pc_centers=num_pc_centers, num_eigenvalues=num_eigenvalues,
                            num_epochs=num_iter)

    # Error metrics
    err_fns = get_err_fns(dset)
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)

        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        with TicToc("EigenPro4 Algorithm"):
            model.fit(Xtr, Ytr, Xts, Yts, err_fns)
        test_model(model, f"EigenPro on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        # print(f"Will train model {flk} on data {dset} with {kfold}-fold CV", flush=True)
        load_fn = get_cv_fn(dset)
        test_errs, train_errs = [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            with TicToc(f"EigenPro4 Algorithm (fold {it})"):
                model.fit(Xtr, Ytr, Xts, Yts, err_fns)

            c_test_errs, c_train_errs = test_model(
                model, f"EigenPro on {dset}", Xts, Yts, Xtr, Ytr, err_fns,
            )
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)

        print_kfold_error_report(kfold, test_errs, train_errs, err_fns)


##############
### BALKON ###
def run_balkon(
    dset: Dataset,
    data_path: str,
    dtype: DataType | None,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    penalty: float,
    kernel: str,
    kfold: int,
    seed: int,
    use_keops: bool,
    block_size: int,
):
    import falkon
    from falkon import kernels
    from falkon.models import balkon
    from falkon.utils import TicToc

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float64
    # Arguments
    if kernel.lower() == "gaussian":
        k = kernels.GaussianKernel(kernel_sigma)
    elif kernel.lower() == "laplacian":
        k = kernels.LaplacianKernel(kernel_sigma)
    elif kernel.lower() == "linear":
        k = kernels.LinearKernel(beta=1.0, gamma=kernel_sigma)
    else:
        raise ValueError(f"Kernel {kernel} not understood for algorithm Falkon")

    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
        cg_tolerance=1e-6,
        cg_stagnation_iterations=4,
        cg_stagnation_threshold=0.96,
        pc_epsilon_32=1e-6,
        pc_epsilon_64=1e-13,
        keops_active="force" if use_keops else "no",
        debug=False,
    )
    flk = balkon.Balkon(
        kernel=k,
        penalty=penalty,
        M=num_centers,
        maxiter=num_iter,
        seed=seed,
        error_fn=None,
        error_every=1,
        options=opt,
        block_size=block_size,
    )

    # Error metrics
    err_fns = get_err_fns(dset)
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        Xtr = Xtr.pin_memory()
        Ytr = Ytr.pin_memory()
        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        with TicToc("BALKON ALGORITHM"):
            flk.error_fn = err_fns[0]
            print(f"Starting to train model {flk} on data {dset}", flush=True)
            flk.fit(Xtr, Ytr, Xts, Yts)
        test_model(flk, f"Balkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        print(f"Will train model {flk} on data {dset} with {kfold}-fold CV", flush=True)
        load_fn = get_cv_fn(dset)
        test_errs, train_errs = [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            with TicToc(f"BALKON ALGORITHM (fold {it})"):
                flk.error_every = err_fns[0]
                flk.fit(Xtr, Ytr, Xts, Yts)
            c_test_errs, c_train_errs = test_model(flk, f"Balkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)

        print_kfold_error_report(kfold, test_errs, train_errs, err_fns)


##############
### FALKON ###
def run_falkon(
    dset: Dataset,
    data_path: str,
    dtype: DataType | None,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    penalty: float,
    kernel: str,
    kfold: int,
    seed: int,
    use_keops: bool,
):
    from falkon import kernels
    from falkon.models import falkon
    from falkon.utils import TicToc

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float64
    # Arguments
    if kernel.lower() == "gaussian":
        k = kernels.GaussianKernel(kernel_sigma)
    elif kernel.lower() == "laplacian":
        k = kernels.LaplacianKernel(kernel_sigma)
    elif kernel.lower() == "linear":
        k = kernels.LinearKernel(beta=1.0, gamma=kernel_sigma)
    else:
        raise ValueError(f"Kernel {kernel} not understood for algorithm Falkon")

    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
        cg_tolerance=1e-6,
        pc_epsilon_32=1e-6,
        pc_epsilon_64=1e-13,
        keops_active="force" if use_keops else "no",
        debug=True,
    )
    flk = falkon.Falkon(
        kernel=k, penalty=penalty, M=num_centers, maxiter=num_iter, seed=seed, error_fn=None, error_every=1, options=opt
    )

    # Error metrics
    err_fns = get_err_fns(dset)
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        Xtr = Xtr.pin_memory()
        Ytr = Ytr.pin_memory()
        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        with TicToc("FALKON ALGORITHM"):
            flk.error_fn = err_fns[0]
            print(f"Starting to train model {flk} on data {dset}", flush=True)
            flk.fit(Xtr, Ytr, Xts, Yts)
        test_model(flk, f"Falkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        print(f"Will train model {flk} on data {dset} with {kfold}-fold CV", flush=True)
        load_fn = get_cv_fn(dset)
        test_errs, train_errs = [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            with TicToc(f"FALKON ALGORITHM (fold {it})"):
                flk.error_every = err_fns[0]
                flk.fit(Xtr, Ytr, Xts, Yts)
            c_test_errs, c_train_errs = test_model(flk, f"Falkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)

        print_kfold_error_report(kfold, test_errs, train_errs, err_fns)


if __name__ == "__main__":
    print("-------------------------------------------")
    print(print(datetime.datetime.now()))
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument("-a", "--algorithm", type=str, choices=["falkon", "balkon", "eigenpro"])
    p.add_argument("-d", "--dataset", type=Dataset, choices=list(Dataset), required=True, help="Dataset")
    p.add_argument("--data-path", type=str, help="Path to dataset")
    p.add_argument(
        "-t",
        "--dtype",
        type=DataType.argparse,
        choices=list(DataType),
        required=False,
        default=None,
        help="Floating point precision to work with. Lower precision will be "
        "faster but less accurate. Certain algorithms require a specific precision. "
        "If this argument is not specified we will use the highest precision "
        "supported by the chosen algorithm.",
    )
    p.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to run the algorithm for.")
    p.add_argument("--subsample", type=int, required=False, default=0, help="Data subsampling")
    p.add_argument("-k", "--kfold", type=int, default=1, help="Number of folds for k-fold CV.")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random number generator seed")
    # Algorithm-specific arguments
    p.add_argument(
        "-M",
        "--num-centers",
        type=int,
        default=0,
        help="Number of Nystroem centers. Used for algorithms falkon, gpytorch and gpflow.",
    )
    p.add_argument(
        "--penalty",
        type=float,
        default=0.0,
        required=False,
        help="Lambda penalty for use in KRR. Needed for the Falkon algorithm.",
    )
    p.add_argument(
        "--sigma", type=float, default=-1.0, required=False, help="Inverse length-scale for the Gaussian kernel."
    )
    p.add_argument(
        "--kernel", type=str, default="gaussian", required=False, help="Type of kernel to use. Used for Falkon"
    )
    p.add_argument("--use-keops", action="store_true", help="Set this flag to enable KeOps.")

    # Balkon-specific
    p.add_argument("--balkon-block-size", type=int, required=False, help="Required for Balkon")

    # EigenPro-specific
    p.add_argument("--epro-pc-centers", type=int, required=False, help="Number of centers used for the preconditioner in EigenPro4")
    p.add_argument("--epro-eigvals", type=int, required=False, help="Number of eigenvalues retained in preconditioner of EigenPro4")

    args = p.parse_args()
    print(f"STARTING {args.algorithm} WITH SEED {args.seed}")

    if args.algorithm == "falkon":
        run_falkon(
            dset=args.dataset,
            data_path=args.data_path,
            use_keops=args.use_keops,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            penalty=args.penalty,
            kernel=args.kernel,
            kfold=args.kfold,
            seed=args.seed,
        )
    elif args.algorithm == "balkon":
        assert args.balkon_block_size is not None
        run_balkon(
            dset=args.dataset,
            data_path=args.data_path,
            use_keops=args.use_keops,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            penalty=args.penalty,
            kernel=args.kernel,
            kfold=args.kfold,
            seed=args.seed,
            block_size=args.balkon_block_size,
        )
    elif args.algorithm == "eigenpro":
        assert args.epro_pc_centers is not None
        assert args.epro_eigvals is not None
        run_eigenpro(
            dset=args.dataset,
            data_path=args.data_path,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            num_pc_centers=args.epro_pc_centers,
            num_eigenvalues=args.epro_eigvals,
            kernel_sigma=args.sigma,
            kernel=args.kernel,
            kfold=args.kfold,
            seed=args.seed,
        )
    else:
        raise ValueError(args.algorithm)
