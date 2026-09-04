
gitimport argparse
import datetime
import functools
import sys
import time
import random

import numpy as np
import torch

from falkon.benchmarks.common.benchmark_utils import Dataset, DataType
from falkon.benchmarks.common.datasets import get_cv_fn, get_load_fn
from falkon.benchmarks.common.error_metrics import get_err_fns
from falkon.benchmarks.models.flk_wrapper import FalkonWrapper

RANDOM_SEED = 123
EIGENPRO_BASE_PATH = "/leonardo/home/userexternal/gmeanti0/EigenPro"
JOKER_BASE_PATH = "./joker/src" #"/leonardo/home/userexternal/gmeanti0/Joker-paper/src"
ASKOTCH_BASE_PATH = "/leonardo/home/userexternal/gmeanti0/fast_krr"


def test_model(model, model_name, Xts, Yts, Xtr, Ytr, err_fns):
    te_pred_time = time.time()
    test_preds = model.predict(Xts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    te_pred_time = time.time() - te_pred_time
    train_preds = None
    if Xtr is not None:
        train_preds = model.predict(Xtr)
    test_errs, train_errs = [], []
    err_names = []
    print(f"Test inference time: {te_pred_time:.2f}s")
    for err_fn in err_fns:
        test_err, test_err_name = err_fn(Yts, test_preds)
        test_errs.append(test_err)
        print(f"Test error {model_name} {test_err_name}: {test_err:9.6f}", flush=True)
        if Xtr is not None and Ytr is not None:
            assert train_preds is not None
            train_err, train_err_name = err_fn(Ytr, train_preds)
            print(f"Train error {model_name} {train_err_name}: {train_err:9.6f}", flush=True)
            train_errs.append(train_err)
        err_names.append(test_err_name)
    return test_errs, train_errs, err_names, te_pred_time


def print_kfold_error_report(k, test_errs, train_errs, err_names, train_times=None, inference_times=None):
    print(f"Full errors: Test {test_errs} - Train {train_errs}")
    print()
    print(f"{k}-Fold Error Report")
    if train_times is not None:
        print(f"Average training time: {np.mean(train_times):.2f}s +- {np.std(train_times):.2f}s")
    if inference_times is not None:
        print(f"Average inference time: {np.mean(inference_times):.2f}s +- {np.std(inference_times):.2f}s")
    for i in range(len(err_names)):
        print(
            f"Final test {err_names[i]}: "
            f"{np.mean([e[i] for e in test_errs]):.6e} +- "
            f"{np.std([e[i] for e in test_errs]):6e}"
        )
        print(
            f"Final train {err_names[i]}: "
            f"{np.mean([e[i] for e in train_errs]):.6e} +- "
            f"{np.std([e[i] for e in train_errs]):.6e}"
        )
        print()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def generic_fit(
    model,
    model_name: str,
    dset: Dataset,
    device: torch.device,
    kfold: int,
    dtype: DataType,
    data_path: str,
    data_on_dev: bool,
    target_class = None
):
    err_fns_ = get_err_fns(dset)
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        if data_on_dev:
            Xtr, Ytr, Xts, Yts = Xtr.to(device), Ytr.to(device), Xts.to(device), Yts.to(device)
        else:
            Xtr = Xtr.pin_memory()
            Ytr = Ytr.pin_memory()

        if target_class is not None:            
            Ytr = Ytr.argmax(-1).to(Xtr.device, Xtr.dtype)
            Yts = Yts.argmax(-1).to(Xtr.device, Xtr.dtype)
            Ytr[Ytr != target_class] = -1.0
            Ytr[Ytr == target_class] = 1.0
            Yts[Yts != target_class] = -1.0
            Yts[Yts == target_class] = 1.0

        model.init_model(Xtr, Ytr, Xts, Yts)
        print(f"Starting to train model {model} on data {dset}", flush=True)

        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns_]
        if hasattr(model, "error_fn"):
            model.error_fn = err_fns[0]
        t_start = time.time()
        model.fit(Xtr, Ytr, Xts, Yts)
        t_elapsed = time.time() - t_start

        if hasattr(model, "fit_times_"):
            train_time = model.fit_times_[-1]
        else:
            train_time = t_elapsed
        test_errs, train_errs, err_names, test_time = test_model(
            model, f"{model_name} on {dset}", Xts, Yts, Xtr, Ytr, err_fns
        )
        print_kfold_error_report(
            1, [test_errs], [train_errs], err_names, train_times=[train_time], inference_times=[test_time]
        )
    else:
        load_fn = get_cv_fn(dset)
        err_names = None
        test_errs, train_errs = [], []
        train_times, test_times = [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns_]
            if hasattr(model, "error_fn"):
                model.error_fn = err_fns[0]

            if data_on_dev:
                Xtr, Ytr, Xts, Yts = Xtr.to(device), Ytr.to(device), Xts.to(device), Yts.to(device)
            else:
                Xtr = Xtr.pin_memory()
                Ytr = Ytr.pin_memory()

            if target_class is not None:            
                Ytr = Ytr.argmax(-1).to(Xtr.device, Xtr.dtype)
                Yts = Yts.argmax(-1).to(Xtr.device, Xtr.dtype)
                Ytr[Ytr != target_class] = -1.0
                Ytr[Ytr == target_class] = 1.0
                Yts[Yts != target_class] = -1.0
                Yts[Yts == target_class] = 1.0

            model.init_model(Xtr, Ytr, Xts, Yts)
            if it == 0:
                print(f"{kfold}-CV training model {model} on data {dset}", flush=True)

            t_start = time.time()
            model.fit(Xtr, Ytr, Xts, Yts)
            t_elapsed = time.time() - t_start

            c_test_errs, c_train_errs, err_names, test_time = test_model(
                model, f"{model_name} on {dset}", Xts, Yts, Xtr, Ytr, err_fns
            )
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            test_times.append(test_time)
            if hasattr(model, "fit_times_"):
                train_times.append(model.fit_times_[-1])
            else:
                train_times.append(t_elapsed)
            if hasattr(model, "reset"):
                model.reset()
            torch.cuda.empty_cache()
        print_kfold_error_report(
            kfold, test_errs, train_errs, err_names, train_times=train_times, inference_times=test_times
        )


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
    sys.path.append(EIGENPRO_BASE_PATH)
    import eigenpro.kernels as kernels  # pyright: ignore[reportMissingImports]
    import eigenpro.utils.device as dev  # pyright: ignore[reportMissingImports]

    from falkon.benchmarks.models.eigenpro_wrapper import EigenProWrapper

    seed_all(seed)

    if dtype is None:
        dtype = DataType.float32
    if kernel == "laplacian":
        kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=kernel_sigma)
    elif kernel == "gaussian":
        kernel_fn = lambda x, z: kernels.gaussian(x, z, bandwidth=kernel_sigma)
    else:
        raise ValueError(kernel)
    device = dev.Device.create(use_gpu_if_available=True)
    model = EigenProWrapper(
        device,
        dtype.to_torch_dtype(),
        kernel_fn,
        num_centers=num_centers,
        num_pc_centers=num_pc_centers,
        num_eigenvalues=num_eigenvalues,
        num_epochs=num_iter,
    )
    pt_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    generic_fit(
        model,
        model_name="EigenPro4",
        dset=dset,
        device=pt_device,
        kfold=kfold,
        dtype=dtype,
        data_path=data_path,
        data_on_dev=False,
    )


##############
### BALKON ###
def run_balkon(
    dset: Dataset,
    data_path: str,
    dtype: DataType | None,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    kernel_nu: float,
    penalty: float,
    kernel: str,
    kfold: int,
    seed: int,
    use_keops: bool,
    block_size: int,
    debug: bool,
):
    import falkon
    from falkon import kernels
    from falkon.models import balkon

    seed_all(seed)

    if dtype is None:
        dtype = DataType.float64
    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=False,#True,
        cg_tolerance=5e-4,
        cg_stagnation_iterations=3,
        cg_stagnation_threshold=0.98,
        pc_epsilon_32=1e-6, # lowered this to 1e-7 for flights (was 1e-6)
        pc_epsilon_64=1e-13,
        keops_active="force" if use_keops else "no",
        keops_sum_scheme="kahan_scheme",
        store_kernel_d_threshold=1500,
        #max_cpu_mem=(160*2**30),
        debug=debug,
    )
    flk = FalkonWrapper(
        balkon.Balkon(
            kernel=kernels.GaussianKernel(1.0),  # placeholder
            penalty=penalty,
            M=num_centers,
            maxiter=num_iter,
            seed=seed,
            error_fn=None,
            error_every=1,
            options=opt,
            block_size=block_size,
        ),
        kernel_type=kernel,
        kernel_sigma=kernel_sigma,
        kernel_nu=kernel_nu,
    )
    pt_device = torch.device('cuda') #torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    generic_fit(
        flk,
        model_name="Balkon",
        dset=dset,
        device=pt_device,
        kfold=kfold,
        dtype=dtype,
        data_path=data_path,
        data_on_dev=False,
    )


def run_askotch(
    dset: Dataset,
    data_path: str,
    dtype : DataType | None,
    task : str, # 'regression' or 'classification'
    kernel_type : str, # 'rbf' or 'matern'
    sigma : float,  # used for both matern and rbf kernel
    lam : float, # regularization
    nu : float,
    rank : int,
    num_iter : int,
    block_size : int,
    target_class : int,
    kfold : int,
    seed : int = 124151
):
    sys.path.append(ASKOTCH_BASE_PATH)
    import pykeops
    import torch
    from pykeops.config import gpu_available

    from falkon.benchmarks.models.askotch_model import ASkotchWrapper
    print(f"{pykeops.__version__=}")
    print(f"{gpu_available=}")
    seed_all(seed)

    pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # type: ignore

    if dtype is None:
        dtype = DataType.float32

    precond_params = {"type": "nystrom", "r": rank, "rho": "damped"}
    askotch = ASkotchWrapper(
        block_size,
        precond_params,
        kernel_type=kernel_type,
        kernel_sigma=sigma,
        kernel_nu=nu,
        unsc_lam=lam,
        task=task, num_iter=num_iter,  target_class=target_class,
        device=pt_device
    )
    generic_fit(
        askotch,
        model_name="ASkotch",
        dset=dset,
        device=pt_device,
        kfold=kfold,
        dtype=dtype,
        data_path=data_path,
        data_on_dev=True,
        target_class=target_class
    )


def run_joker(
    dset: Dataset,
    data_path: str,
    dtype : DataType | None,
    num_iter_subprob : int,
    criterion : str,
    c : float, # penalty parameter of error term
    kernel_type : str, # 'rbf' or 'matern'
    sigma : float,  # used for both matern and rbf kernel
    num_iter : int,
    block_size : int,
    data_block_size : int,
    max_region_size : float,
    opt_name : str, #trust_region or tncg
    kfold : int,
    incore : bool,
    nrff : int,
    n_fastfood : int,
    inexact_type : str,
    region_shrink_freq : int = 1000,
    region_shrink_rate : float = 0.5,
    delta_huber : float = -1.0,
    eps : float = -1.0,
    seed : int = 124151
):
    sys.path.append(JOKER_BASE_PATH)
    from criterion import make_criterion # pyright: ignore[reportMissingImports]

    from falkon.benchmarks.models.joker_model import JokerWrapper

    seed_all(seed)

    pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = DataType.float32

    crit = make_criterion(
        criterion, c=c, delta=delta_huber, eps=eps, dtype=dtype.to_torch_dtype()
    )
    if kernel_type == "laplacian":
        joker_ktype = "lap"
    elif kernel_type == "gaussian":
        joker_ktype = "rbf"
    else:
        raise ValueError(f"Unrecognized kernel for Joker: {kernel_type}")
    model = JokerWrapper(
        inexact_type=inexact_type,
        dtype=dtype.to_torch_dtype(),
        crit=crit,
        kernel_type=joker_ktype,
        kernel_sigma=sigma,
        device=pt_device,
        nrff=nrff,
        nfastfood=n_fastfood,
        block_size=block_size,
        incore=incore,
        data_block_size=data_block_size,
        opt_name=opt_name,
        num_iter=num_iter,
        num_iter_subprob=num_iter_subprob,
        max_region_size=max_region_size,
        region_shrink_freq=region_shrink_freq,
        region_shrink_rate=region_shrink_rate,
    )
    generic_fit(
        model,
        model_name="Joker",
        dset=dset,
        device=pt_device,
        kfold=kfold,
        dtype=dtype,
        data_path=data_path,
        data_on_dev=False,
    )


def run_falkon(
    dset: Dataset,
    data_path: str,
    dtype: DataType | None,
    num_iter: int,
    num_centers: int,
    kernel_sigma: float,
    kernel_nu: float,
    penalty: float,
    kernel: str,
    kfold: int,
    seed: int,
    use_keops: bool,
    pos_weight: float | None,
    debug: bool,
):
    from falkon import kernels
    from falkon.models import falkon

    seed_all(seed)

    if dtype is None:
        dtype = DataType.float64
    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=False,
        cg_tolerance=5e-4,
        cg_stagnation_iterations=3,
        cg_stagnation_threshold=0.98,
        pc_epsilon_32=1e-6,
        pc_epsilon_64=1e-13,
        keops_active="force" if use_keops else "no",
        keops_sum_scheme="kahan_scheme",
        store_kernel_d_threshold=1500,
        #max_cpu_mem=(160*2**30),
        debug=debug,
    )
    weight_fn = None
    if pos_weight is not None:
        neg_weight = 1.0
        weight_fn = lambda Y, X, indices: torch.where(Y < 0, neg_weight, pos_weight)

    flk = FalkonWrapper(
        falkon.Falkon(
            kernel=kernels.GaussianKernel(1.0),  # placeholder
            penalty=penalty,
            M=num_centers,
            maxiter=num_iter,
            seed=seed,
            error_fn=None,
            error_every=1,
            weight_fn=weight_fn,
            options=opt,
        ),
        kernel_type=kernel,
        kernel_sigma=kernel_sigma,
        kernel_nu=kernel_nu,
    )
    pt_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    generic_fit(
        flk,
        model_name="Falkon",
        dset=dset,
        device=pt_device,
        kfold=kfold,
        dtype=dtype,
        data_path=data_path,
        data_on_dev=False,
    )


if __name__ == "__main__":
    print("-------------------------------------------")
    print(print(datetime.datetime.now()))
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument("-a", "--algorithm", type=str, choices=["falkon", "balkon", "eigenpro", "askotch", "joker"])
    p.add_argument("-d", "--dataset", type=Dataset, choices=list(Dataset), required=True, help="Dataset")
    p.add_argument("--data-path", type=str, help="Path to dataset")
    p.add_argument(
        "-t",
        "--dtype",
        type=DataType.argparse,
        choices=list(DataType),
        required=False,
        default=None,
        help="Floating point precision to work with."
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
    p.add_argument("--debug", action="store_true")

    # Falkon-specific
    p.add_argument("--falkon-pos-weight", type=float, required=False, help="Positive-class weight for falkon classification problem")

    # Balkon-specific
    p.add_argument("--balkon-block-size", type=int, required=False, help="Required for Balkon")

    # EigenPro-specific
    p.add_argument(
        "--epro-pc-centers", type=int, required=False, help="Number of centers used for the preconditioner in EigenPro4"
    )
    p.add_argument(
        "--epro-eigvals", type=int, required=False, help="Number of eigenvalues retained in preconditioner of EigenPro4"
    )

    ######### ASKOTCH PARAMS ##############
    p.add_argument('--askotch-task', default='classification', choices=['classification', 'regression'], help='Task tackled by ASkotch')
    p.add_argument('--askotch-bs', default=100, type=int, help='Block-size used in ASkotch')
    p.add_argument('--askotch-target-class', default=None, type=int, help='Target class (for multiclass classification)')

    p.add_argument('--nu', default=5/2, type=float, help='nu of Matern kernel')

    ########## JOKER PARAMS
    p.add_argument('--joker-criterion', type=str, default=None, choices=["mse", "huber", "svm", "log", "svr"], help="Which criterion to use for Joker")
    p.add_argument('--joker-c', type=float, default=1.0, help='Penalty parameter C of the error term in Joker')
    p.add_argument('--joker-delta-huber', type=float, default=-1, help='Joker\'s parameter delta')
    p.add_argument('--joker-eps', type=float, default=-1, help='Joker\'s parameter epsilon')
    p.add_argument("--joker-blksz", type=int, default=512, help="The block size for optimization")
    p.add_argument('--joker-incore', action='store_true', help='Set this to place the data into GPU (Joker)')
    p.add_argument('--joker-inexact-type', type=str, default='rff', choices=['fastfood', 'rff', 'no'], help='Type of inexactness (Joker)')
    p.add_argument('--joker-nrff', type=int, default=50000, help="Number of samples for RFF approximation")
    p.add_argument("--joker-n-fastfood", type=int, default=100, help="Number of samples for Fastfood approximation")

    args = p.parse_args()
    print(f"STARTING {args.algorithm} WITH SEED {args.seed}. K-fold={args.kfold}")

    if args.algorithm == "falkon":
        run_falkon(
            dset=args.dataset,
            data_path=args.data_path,
            use_keops=args.use_keops,
            dtype=args.dtype,
            num_iter=args.epochs,
            num_centers=args.num_centers,
            kernel_sigma=args.sigma,
            kernel_nu=args.nu,
            penalty=args.penalty,
            kernel=args.kernel,
            pos_weight=args.falkon_pos_weight,
            kfold=args.kfold,
            seed=args.seed,
            debug=args.debug,
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
            kernel_nu=args.nu,
            penalty=args.penalty,
            kernel=args.kernel,
            kfold=args.kfold,
            seed=args.seed,
            block_size=args.balkon_block_size,
            debug=args.debug,
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
    elif args.algorithm == "askotch":
        run_askotch(
            dset = args.dataset,
            data_path=args.data_path,
            dtype=args.dtype,
            task=args.askotch_task,
            sigma=args.sigma,
            nu=args.nu,
            lam=args.penalty,
            kernel_type = args.kernel,
            rank=args.num_centers,
            num_iter=args.epochs,
            block_size=args.askotch_bs,
            target_class=args.askotch_target_class,
            kfold=args.kfold,
            seed=args.seed
        )
    elif args.algorithm == "joker":
        assert args.joker_criterion is not None
        run_joker(
            dset = args.dataset,
            data_path=args.data_path,
            dtype=args.dtype,
            num_iter_subprob=50,
            criterion=args.joker_criterion,
            c=args.joker_c,
            data_block_size=args.joker_blksz,
            block_size=args.joker_blksz,
            sigma=args.sigma,
            max_region_size=64,
            opt_name = "trust_region",
            kernel_type = args.kernel,
            region_shrink_freq = 1000,
            region_shrink_rate = 0.5,
            num_iter=args.epochs,
            delta_huber = args.joker_delta_huber,
            eps = args.joker_eps,
            inexact_type = args.joker_inexact_type,
            n_fastfood = args.joker_n_fastfood,
            nrff=args.joker_nrff,
            incore = args.joker_incore,
            kfold=args.kfold,
            seed=args.seed
        )
    else:
        raise ValueError(args.algorithm)
