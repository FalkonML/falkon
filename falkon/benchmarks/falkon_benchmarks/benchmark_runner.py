import argparse
import functools
import sys
import time
from typing import List, Optional

import numpy as np

from falkon.benchmarks.common.benchmark_utils import Algorithm, Dataset, DataType, VariationalDistribution
from falkon.benchmarks.common.datasets import get_cv_fn, get_load_fn
from falkon.benchmarks.common.error_metrics import get_err_fns, get_tf_err_fn

RANDOM_SEED = 123
EPRO_DIRECTORY = "../../EigenPro2"


def test_model(model, model_name, Xts, Yts, Xtr, Ytr, err_fns):
    test_preds = model.predict(Xts)
    if Xtr is not None:
        train_preds = model.predict(Xtr)
    test_errs, train_errs = [], []
    for err_fn in err_fns:
        test_err, test_err_name = err_fn(Yts, test_preds)
        test_errs.append(test_err)
        print(f"Test {model_name} {test_err_name}: {test_err:9.6f}", flush=True)
        if Xtr is not None and Ytr is not None:
            train_err, train_err_name = err_fn(Ytr, train_preds)
            print(f"Train {model_name} {train_err_name}: {train_err:9.6f}", flush=True)
            train_errs.append(train_err)
    return test_errs, train_errs


def run_epro(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    num_iter: int,
    kernel_sigma: float,
    n_subsample: Optional[int],
    data_subsample: Optional[int],
    q: Optional[int],
    kfold: int,
    eta_divisor: int,
    seed: int,
):
    sys.path.append(EPRO_DIRECTORY)
    import kernels
    import tensorflow as tf
    from eigenpro import EigenPro

    tf.set_random_seed(seed)
    np.random.seed(seed)

    if dtype is None:
        dtype = DataType.float32
    if dtype.to_numpy_dtype() != np.float32:
        raise RuntimeError("EigenPro can only run on single-precision floats.")

    # Error metrics
    err_fns = get_err_fns(dset)
    tf_err_fn = get_tf_err_fn(dset)

    # Create kernel
    kernel = functools.partial(kernels.GaussianKernel, sigma=kernel_sigma)

    # Additional fixed params
    mem_gb = 11
    print(
        f"Starting EigenPro solver with {n_subsample} subsamples, "
        f"{q}-top eigensystem, {eta_divisor} eta-divisor. Random seed: {seed}"
    )
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=False)
        if data_subsample is not None:
            Xtr = Xtr[:data_subsample]
            Ytr = Ytr[:data_subsample]
            print(f"SUBSAMPLED INPUT DATA TO {Xtr.shape[0]} TRAINING SAMPLES", flush=True)
        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        tf_err_fn = functools.partial(tf_err_fn, **kwargs)
        tf_err_fn.__name__ = "tf_error"
        model = EigenPro(
            kernel,
            Xtr,
            n_label=Ytr.shape[1],
            mem_gb=mem_gb,
            n_subsample=n_subsample,
            q=q,
            bs=None,
            metric=tf_err_fn,
            seed=seed,
            eta_divisor=eta_divisor,
        )
        print(f"Starting to train model {model} on data {dset}", flush=True)
        t_s = time.time()
        model.fit(Xtr, Ytr, x_val=Xts, y_val=Yts, epochs=np.arange(num_iter - 1) + 1)
        print(f"Training of algorithm {algorithm} on {dset} done in {time.time() - t_s:.2f}s", flush=True)
        test_model(model, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        print(f"Will train EigenPro model on data {dset} with {kfold}-fold CV", flush=True)
        load_fn = get_cv_fn(dset)
        iteration = 0
        test_errs, train_errs = [], []

        for Xtr, Ytr, Xts, Yts, kwargs in load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=False):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            tf_err_fn = functools.partial(tf_err_fn, **kwargs)
            tf_err_fn.__name__ = "tf_error"
            model = EigenPro(
                kernel,
                Xtr,
                n_label=Ytr.shape[1],
                mem_gb=mem_gb,
                n_subsample=n_subsample,
                q=q,
                bs=None,
                metric=tf_err_fn,
                seed=seed,
            )
            print(f"Starting EPRO fit (fold {iteration})")
            model.fit(Xtr, Ytr, x_val=Xts, y_val=Yts, epochs=np.arange(num_iter - 1) + 1)
            iteration += 1
            c_test_errs, c_train_errs = test_model(model, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)

        print(f"Full errors: Test {test_errs} - Train {train_errs}")
        print()
        print(f"{kfold}-Fold Error Report")
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


def run_gpytorch_sgpr(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    lr: float,
    num_iter: int,
    num_centers: int,
    learn_ind_pts: bool,
    seed: int,
):
    import torch

    from falkon.benchmarks.models.gpytorch_sgpr import GpytorchSGPR

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float32
    if dtype.to_numpy_dtype() != np.float32:
        raise RuntimeError(f"{algorithm} can only run on single-precision floats.")
    # Error metrics
    err_fns = get_err_fns(dset)

    # Load data
    load_fn = get_load_fn(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
    if Ytr.shape[1] != 1:
        raise NotImplementedError("GPyTorch SGPR only implemented for single-output problems.")
    err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]

    # Extract inducing points at random
    inducing_idx = np.random.choice(Xtr.shape[0], num_centers, replace=False)
    inducing_points = Xtr[inducing_idx].reshape(num_centers, -1)

    # Initialize model class
    model = GpytorchSGPR(
        inducing_points, err_fns[0], num_epochs=num_iter, use_cuda=True, lr=lr, learn_ind_pts=learn_ind_pts
    )

    # Initialize training
    print(f"Starting to train model {model} on data {dset}", flush=True)
    t_s = time.time()
    model.do_train(Xtr, Ytr, Xts, Yts)
    print(f"Training of {algorithm} on {dset} complete in {time.time() - t_s:.2f}s", flush=True)
    # if isinstance(model, TwoClassVGP):
    #     # Need Ys in range [0,1] for correct error calculation
    #     Yts = (Yts + 1) / 2
    #     Ytr = (Ytr + 1) / 2
    test_model(model, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)


def run_gpytorch(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    batch_size: int,
    lr: float,
    natgrad_lr: float,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    var_dist: str,
    learn_ind_pts: bool,
    kfold: int,
    seed: int,
    ind_pt_file: Optional[str] = None,
):
    import gpytorch
    import torch

    from falkon.benchmarks.models.gpytorch_variational_models import MultiClassVGP, RegressionVGP, TwoClassVGP

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float32
    if dtype.to_numpy_dtype() != np.float32:
        raise RuntimeError(f"{algorithm} can only run on single-precision floats.")
    # Error metrics
    err_fns = get_err_fns(dset)

    def get_model(Xtr, num_outputs, err_fn):
        num_samples = Xtr.shape[0]
        # Inducing points
        inducing_idx = np.random.choice(num_samples, num_centers, replace=False)
        inducing_points = Xtr[inducing_idx].reshape(num_centers, -1)
        print("Took %d random inducing points" % (inducing_points.shape[0]))
        # Kernel
        if num_outputs == 1:
            # Kernel has 1 length-scale!
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=None))
            kernel.base_kernel.lengthscale = kernel_sigma
            # kernel = gpytorch.kernels.keops.RBFKernel(ard_num_dims=None)
            # kernel.lengthscale = kernel_sigma
        else:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=None, batch_shape=torch.Size([num_outputs]))
            )
        if algorithm == Algorithm.GPYTORCH_CLS:
            if num_outputs == 1:
                # 2 classes
                model = TwoClassVGP(
                    inducing_points,
                    kernel,
                    var_dist=var_dist,
                    err_fn=err_fn,
                    mb_size=batch_size,
                    num_data=num_samples,
                    num_epochs=num_iter,
                    use_cuda=True,
                    lr=lr,
                    natgrad_lr=natgrad_lr,
                    learn_ind_pts=learn_ind_pts,
                )
            else:
                # multiclass
                model = MultiClassVGP(
                    inducing_points,
                    kernel,
                    num_classes=num_outputs,
                    var_dist=var_dist,
                    err_fn=err_fn,
                    mb_size=batch_size,
                    num_data=num_samples,
                    num_epochs=num_iter,
                    use_cuda=True,
                    natgrad_lr=natgrad_lr,
                    lr=lr,
                    learn_ind_pts=learn_ind_pts,
                )
        else:
            if num_outputs != 1:
                raise NotImplementedError("Multi-output regression not yet implemented.")
            model = RegressionVGP(
                inducing_points,
                kernel,
                var_dist=var_dist,
                err_fn=err_fn,
                mb_size=batch_size,
                num_data=num_samples,
                num_epochs=num_iter,
                use_cuda=True,
                natgrad_lr=natgrad_lr,
                lr=lr,
                learn_ind_pts=learn_ind_pts,
            )
        return model

    # Load data
    load_fn = get_load_fn(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
    err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
    model = get_model(Xtr, Ytr.shape[1], err_fns[0])
    print(f"Starting to train model {model} on data {dset}", flush=True)
    t_s = time.time()
    with gpytorch.settings.fast_computations(False, False, False):
        model.do_train(Xtr, Ytr, Xts, Yts)
    print(f"Training of {algorithm} on {dset} complete in {time.time() - t_s:.2f}s", flush=True)
    if isinstance(model, TwoClassVGP):
        # Need Ys in range [0,1] for correct error calculation
        Yts = (Yts + 1) / 2
        Ytr = (Ytr + 1) / 2
    with gpytorch.settings.fast_computations(False, False, False):
        test_model(model, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)


def run_falkon(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    penalty: float,
    kernel: str,
    kfold: int,
    seed: int,
):
    import torch

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
        raise ValueError(f"Kernel {kernel} not understood for algorithm {algorithm}")

    opt = falkon.FalkonOptions(
        compute_arch_speed=False, no_single_kernel=True, pc_epsilon_32=1e-6, pc_epsilon_64=1e-13, debug=True
    )
    flk = falkon.Falkon(
        kernel=k, penalty=penalty, M=num_centers, maxiter=num_iter, seed=seed, error_fn=None, error_every=1, options=opt
    )

    # Error metrics
    err_fns = get_err_fns(dset)
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
        Xtr = Xtr.pin_memory()
        Ytr = Ytr.pin_memory()
        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        with TicToc("FALKON ALGORITHM"):
            flk.error_fn = err_fns[0]
            print(f"Starting to train model {flk} on data {dset}", flush=True)
            flk.fit(Xtr, Ytr, Xts, Yts)
        test_model(flk, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        print(f"Will train model {flk} on data {dset} with {kfold}-fold CV", flush=True)
        load_fn = get_cv_fn(dset)
        iteration = 0
        test_errs, train_errs = [], []

        for Xtr, Ytr, Xts, Yts, kwargs in load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            with TicToc(f"FALKON ALGORITHM (fold {iteration})"):
                flk.error_every = err_fns[0]
                flk.fit(Xtr, Ytr, Xts, Yts)
            iteration += 1
            c_test_errs, c_train_errs = test_model(flk, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)

        print(f"Full errors: Test {test_errs} - Train {train_errs}")
        print()
        print(f"{kfold}-Fold Error Report")
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


def run_logistic_falkon(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    iter_list: List[int],
    penalty_list: List[float],
    num_centers: int,
    kernel_sigma: float,
    kernel: str,
    seed: int,
):
    import torch

    import falkon
    from falkon import kernels
    from falkon.gsc_losses import LogisticLoss
    from falkon.models import logistic_falkon
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
        raise ValueError(f"Kernel {kernel} not understood for algorithm {algorithm}")
    opt = falkon.FalkonOptions(
        compute_arch_speed=False, no_single_kernel=True, pc_epsilon_32=1e-6, pc_epsilon_64=1e-13, debug=True
    )
    loss = LogisticLoss(kernel=k)
    flk = logistic_falkon.LogisticFalkon(
        kernel=k,
        loss=loss,
        penalty_list=penalty_list,
        iter_list=iter_list,
        M=num_centers,
        seed=seed,
        error_fn=None,
        error_every=1,
        options=opt,
    )

    # Error metrics
    err_fns = get_err_fns(dset)
    # Load data
    load_fn = get_load_fn(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True)
    Xtr = Xtr.pin_memory()
    Ytr = Ytr.pin_memory()
    err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
    with TicToc("LOGISTIC FALKON ALGORITHM"):
        flk.error_fn = err_fns[0]
        print(f"Starting to train model {flk} on data {dset}", flush=True)
        flk.fit(Xtr, Ytr, Xts, Yts)
    test_model(flk, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)


def run_sgpr_gpflow(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    lr: float,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    learn_ind_pts: bool,
    kernel_variance: float,
    seed: int,
):
    import gpflow
    import tensorflow as tf

    from falkon.benchmarks.models.gpflow_model import TrainableSGPR

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float32
    if dtype == DataType.float32:
        gpflow.config.set_default_float(np.float32)

    err_fns = get_err_fns(dset)

    # Kernel
    sigma_initial = np.array(kernel_sigma, dtype=dtype.to_numpy_dtype())
    kernel = gpflow.kernels.SquaredExponential(lengthscales=sigma_initial, variance=kernel_variance)

    # Data loading
    load_fn = get_load_fn(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=False, as_tf=True)
    err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]

    # Inducing points
    inducing_idx = np.random.choice(Xtr.shape[0], num_centers, replace=False)
    inducing_points = Xtr[inducing_idx].reshape(num_centers, -1)
    print("Took %d random inducing points" % (inducing_points.shape[0]))
    if Ytr.shape[1] != 1:
        raise NotImplementedError("SGPR GPFLOW only implemented for 1 output")

    # Define model, train and test
    model = TrainableSGPR(
        kernel=kernel,
        inducing_points=inducing_points,
        num_iter=num_iter,
        err_fn=err_fns[0],
        train_hyperparams=learn_ind_pts,
        lr=lr,
    )
    t_s = time.time()
    print(f"Starting to train model {model} on data {dset}", flush=True)
    model.fit(Xtr, Ytr, Xts, Yts)
    print(f"Training of {algorithm} on {dset} complete in {time.time() - t_s:.2f}s", flush=True)
    test_model(model, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)


def run_gpflow(
    dset: Dataset,
    algorithm: Algorithm,
    dtype: Optional[DataType],
    batch_size: int,
    lr: float,
    natgrad_lr: float,
    var_dist: str,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    learn_ind_pts: bool,
    error_every: int,
    kernel_variance: float,
    kfold: int,
    seed: int,
    ind_pt_file: Optional[str] = None,
):
    import gpflow
    import tensorflow as tf

    from falkon.benchmarks.models.gpflow_model import TrainableSVGP

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Data types
    if dtype is None:
        dtype = DataType.float32
    if dtype == DataType.float32:
        gpflow.config.set_default_float(np.float32)

    err_fns = get_err_fns(dset)

    def get_model(Xtr, num_outputs, err_fn, kernel):
        # Inducing points
        inducing_idx = np.random.choice(Xtr.shape[0], num_centers, replace=False)
        inducing_points = Xtr[inducing_idx].reshape(num_centers, -1)
        print("Took %d random inducing points" % (inducing_points.shape[0]))

        num_classes = 0
        if algorithm == Algorithm.GPFLOW_CLS:
            if num_outputs == 1:
                num_classes = 2
            else:
                num_classes = num_outputs
        model = TrainableSVGP(
            kernel=kernel,
            inducing_points=inducing_points,
            batch_size=batch_size,
            num_iter=num_iter,
            err_fn=err_fn,
            classif=num_classes,
            lr=lr,
            var_dist=var_dist,
            error_every=error_every,
            train_hyperparams=learn_ind_pts,
            optimize_centers=True,
            natgrad_lr=natgrad_lr,
        )
        return model

    load_fn = get_load_fn(dset)
    Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=False, as_tf=True)
    err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
    # Kernel
    sigma_initial = np.array(kernel_sigma, dtype=dtype.to_numpy_dtype())
    kernel = gpflow.kernels.SquaredExponential(lengthscales=sigma_initial, variance=kernel_variance)
    model = get_model(Xtr, Ytr.shape[1], err_fns[0], kernel)
    t_s = time.time()
    print(f"Starting to train model {model} on data {dset}", flush=True)
    model.fit(Xtr, Ytr, Xts, Yts)
    print(f"Training of {algorithm} on {dset} complete in {time.time() - t_s:.2f}s", flush=True)
    if model.num_classes == 2:
        Yts = (Yts + 1) / 2
        Ytr = (Ytr + 1) / 2
    test_model(model, f"{algorithm} on {dset}", Xts, Yts, Xtr, Ytr, err_fns)


if __name__ == "__main__":
    import datetime

    print("-------------------------------------------")
    print(print(datetime.datetime.now()))
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument(
        "-a",
        "--algorithm",
        type=Algorithm,
        choices=list(Algorithm),
        required=True,
        help="The algorithm which should be used for predictions.",
    )
    p.add_argument("-d", "--dataset", type=Dataset, choices=list(Dataset), required=True, help="Dataset")
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
        help="Number of Nystroem centers. Used for algorithms " "falkon, gpytorch and gpflow.",
    )

    p.add_argument("--natgrad-lr", type=float, default=0.0001, help="Natural gradient learning rate (GPFlow)")

    p.add_argument(
        "--var-dist",
        type=VariationalDistribution,
        default=None,
        required=False,
        help="Form of the variational distribution used in GPytorch",
    )
    p.add_argument("--learn-hyperparams", action="store_true", help="Whether gpytorch should learn hyperparameters")
    p.add_argument(
        "--inducing-point-file", type=str, default=None, required=False, help="file with saved inducing points"
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
        "--batch-size",
        type=int,
        default=4096,
        required=False,
        help="Mini-batch size to be used for stochastic methods (GPytorch)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.001,
        required=False,
        help="Learning rate, used for only certain algorithms (GPytorch)",
    )
    p.add_argument(
        "--n-subsample",
        type=int,
        default=None,
        required=False,
        help="Number of samples to be used for the EigenPro SVD preconditioner",
    )
    p.add_argument(
        "--data-subsample",
        type=int,
        default=None,
        required=False,
        help="Subsample the input data to this number of samples (EigenPro)",
    )
    p.add_argument(
        "--epro-q", type=int, default=None, required=False, help="Top-q eigenvalues to take for eigenpro preconditioner"
    )
    p.add_argument(
        "--kernel", type=str, default="gaussian", required=False, help="Type of kernel to use. Used for Falkon"
    )
    p.add_argument(
        "--error-every", type=int, default=1000, required=False, help="How often to display validation error (GPFlow)"
    )
    p.add_argument(
        "--kernel-variance",
        type=float,
        default=1.0,
        required=False,
        help="Default kernel variance for GPFlow RBF kernel",
    )
    p.add_argument(
        "--eta-divisor", type=float, default=1.0, required=False, help="Learning-rate regulator for EigenPro"
    )
    p.add_argument(
        "--iter-list", type=int, nargs="*", default=[], required=False, help="List of CG iterations for logistic falkon"
    )
    p.add_argument(
        "--penalty-list",
        type=float,
        nargs="*",
        default=[],
        required=False,
        help="List of penalty values for logistic falkon",
    )

    args = p.parse_args()
    print(f"STARTING WITH SEED {args.seed}")

    if args.algorithm == Algorithm.FALKON:
        run_falkon(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            penalty=args.penalty,
            kernel=args.kernel,
            kfold=args.kfold,
            seed=args.seed,
        )
    elif args.algorithm == Algorithm.LOGISTIC_FALKON:
        run_logistic_falkon(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            iter_list=args.iter_list,
            penalty_list=args.penalty_list,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            kernel=args.kernel,
            seed=args.seed,
        )
    elif args.algorithm == Algorithm.EIGENPRO:
        run_epro(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            num_iter=args.epochs,
            kernel_sigma=args.sigma,
            n_subsample=args.n_subsample,
            q=args.epro_q,
            kfold=args.kfold,
            seed=args.seed,
            data_subsample=args.data_subsample,
            eta_divisor=args.eta_divisor,
        )
    elif args.algorithm in {Algorithm.GPYTORCH_CLS, Algorithm.GPYTORCH_REG}:
        run_gpytorch(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            var_dist=str(args.var_dist),
            batch_size=args.batch_size,
            lr=args.lr,
            learn_ind_pts=args.learn_hyperparams,
            ind_pt_file=args.inducing_point_file,
            kfold=args.kfold,
            seed=args.seed,
            natgrad_lr=args.natgrad_lr,
        )
    elif args.algorithm in {Algorithm.GPFLOW_CLS, Algorithm.GPFLOW_REG}:
        run_gpflow(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            var_dist=str(args.var_dist),
            batch_size=args.batch_size,
            lr=args.lr,
            natgrad_lr=args.natgrad_lr,
            learn_ind_pts=args.learn_hyperparams,
            ind_pt_file=args.inducing_point_file,
            error_every=args.error_every,
            kernel_variance=args.kernel_variance,
            kfold=args.kfold,
            seed=args.seed,
        )
    elif args.algorithm == Algorithm.GPYTORCH_SGPR:
        run_gpytorch_sgpr(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            lr=args.lr,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            learn_ind_pts=args.learn_hyperparams,
            seed=args.seed,
        )
    elif args.algorithm == Algorithm.GPFLOW_SGPR:
        run_sgpr_gpflow(
            dset=args.dataset,
            algorithm=args.algorithm,
            dtype=args.dtype,
            lr=args.lr,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            learn_ind_pts=args.learn_hyperparams,
            kernel_variance=args.kernel_variance,
            seed=args.seed,
        )
    else:
        raise NotImplementedError(f"No benchmark implemented for algorithm {args.algorithm}.")
