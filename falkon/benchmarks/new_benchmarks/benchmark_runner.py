import argparse
import datetime
import functools
import sys
import time

import numpy as np
import torch

from falkon.benchmarks.common.benchmark_utils import Dataset, DataType
from falkon.benchmarks.common.datasets import get_cv_fn, get_load_fn
from falkon.benchmarks.common.error_metrics import get_err_fns


RANDOM_SEED = 123
EIGENPRO_BASE_PATH = "/leonardo/home/userexternal/gmeanti0/EigenPro"
JOKER_BASE_PATH = "/leonardo/home/userexternal/gmeanti0/Joker-paper/src"
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
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_median_sigma_square(data, num_samples=10000):
    sub_data = data[:num_samples]
    return torch.median(torch.pdist(sub_data) ** 2)

def get_median_sigma(data, num_samples=10000):
    sub_data = data[:num_samples]
    return torch.median(torch.pdist(sub_data))


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
    import eigenpro.kernels as kernels  # pyright: ignore[reportMissingImports]
    import eigenpro.utils.device as dev  # pyright: ignore[reportMissingImports]

    from falkon.benchmarks.models.eigenpro_wrapper import EigenProWrapper
    
    seed_all(seed)

    data_dtype = DataType.float32
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

    # Error metrics
    err_fns = get_err_fns(dset)
    if kfold == 1:
        # Load data
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=data_dtype.to_numpy_dtype(), as_torch=True, path=data_path)

        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        with TicToc("EigenPro4 Algorithm"):
            model.fit(Xtr, Ytr, Xts, Yts, err_fns)
        test_errs, train_errs, err_names, test_time = test_model(model, f"EigenPro on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        # print(f"Will train model {flk} on data {dset} with {kfold}-fold CV", flush=True)
        load_fn = get_cv_fn(dset)
        err_names = None
        test_errs, train_errs = [], []
        train_times, test_times = [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=data_dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            with TicToc(f"EigenPro4 Algorithm (fold {it})"):
                model.fit(Xtr, Ytr, Xts, Yts, err_fns)

            c_test_errs, c_train_errs, err_names, test_time = test_model(
                model, f"EigenPro on {dset}", Xts, Yts, Xtr, Ytr, err_fns
            )
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            test_times.append(test_time)
            train_times.append(model.epoch_times[-1])
            model.reset()

        print_kfold_error_report(kfold, test_errs, train_errs, err_names, train_times=train_times, inference_times=test_times)


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
    debug: bool,
):
    import falkon
    from falkon import kernels
    from falkon.models import balkon
    from falkon.utils import TicToc

    seed_all(seed)

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
        raise ValueError(f"Kernel {kernel} not understood for algorithm Balkon")

    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
        cg_tolerance=2e-7,
        cg_stagnation_iterations=3,
        cg_stagnation_threshold=0.98,
        pc_epsilon_32=1e-7, # lowered this for flights (was 1e-6)
        pc_epsilon_64=1e-13,
        keops_active="force" if use_keops else "no",
        store_kernel_d_threshold=1500,
        #max_cpu_mem=(160*2**30),
        debug=debug,
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
        test_errs, train_errs, err_names, test_time = test_model(flk, f"Balkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
    else:
        print(f"{kfold}-CV training model {flk} on data {dset}", flush=True)
        load_fn = get_cv_fn(dset)
        test_errs, train_errs = [], []
        test_times, train_times = [], []
        err_names = None

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            with TicToc(f"BALKON ALGORITHM (fold {it})"):
                flk.error_fn = err_fns[0]
                flk.fit(Xtr, Ytr, Xts, Yts)
            train_times.append(flk.fit_times_[-1])
            c_test_errs, c_train_errs, err_names, c_test_time = test_model(flk, f"Balkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            test_times.append(c_test_time)

        print_kfold_error_report(kfold, test_errs, train_errs, err_names, train_times=train_times, inference_times=test_times)


def run_askotch(
    dset: Dataset,
    data_path: str,
    dtype : DataType | None,
    task : str, # 'regression' or 'classification'
    kernel_type : str, # 'rbf' or 'matern'
    sigma : float,  # used for both matern and rbf kernel
    lam : float, # regularization
    rank : int,
    num_iter : int,
    block_size : int,
    kfold : int,
    device : str = 'cuda',
    seed : int = 124151
):
    sys.path.append(ASKOTCH_BASE_PATH)
    import torch
    import pykeops
    from pykeops.config import gpu_available
    from falkon.benchmarks.models.askotch_model import ASkotchWrapper
    print(f"{pykeops.__version__=}")
    print(f"{gpu_available=}")
    seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # type: ignore

    if dtype is None:
        dtype = DataType.float32
#    if dtype.to_numpy_dtype() != np.float32:
#        raise RuntimeError(f"ASkotch can only run on single-precision floats.")

    err_fns = get_err_fns(dset)
    
    def get_kernel_params(X, sigma):
        if sigma < 0:
            sigma = get_median_sigma(X).item()
            print(f"kernel sigma chosen with median heuristic: {sigma}")
        if kernel_type == 'gaussian':
            return {'type' : 'rbf', 'sigma': sigma}
        elif kernel_type == 'laplacian':
            return {'type' : 'l1_laplace', 'sigma': sigma}
        else:
            raise ValueError(f"Kernel {kernel_type} not valid for ASkotch")
    
    precond_params = {"type": "nystrom", "r": rank, "rho": "damped"}
    
    if kfold == 1:
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        Xtr_as, Ytr_as, Xts_as, Yts_as = Xtr.to(device), Ytr.to(device).flatten(), Xts.to(device), Yts.to(device).flatten()
        if block_size <= 0:
            block_size = Xtr.shape[0] // 100
        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        kernel_params = get_kernel_params(Xtr, sigma)
        wrapper = ASkotchWrapper(
            Xtr_as, Ytr_as, Xts_as, Yts_as, block_size, precond_params, kernel_params, lam*Xtr_as.shape[0], task, num_iter, device
        )

        tr_time = time.time()
        wrapper.fit(Xtr_as, Ytr_as, Xts_as, Yts_as, err_fns[0])
        tr_time = time.time() - tr_time

        te_err, tr_err, err_names, te_pred_time = test_model(wrapper, f"ASkotch on {dset}", Xts.to(device), Yts.to(device), Xtr.to(device), Ytr.to(device), err_fns)
        print(f"ASkotch timings. training={tr_time:.2f}s inference={te_pred_time:.2f}s")
    else:
        load_fn = get_cv_fn(dset)
        err_names = None
        test_errs, train_errs, train_times, test_pred_times = [], [], [], []
        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            Xtr_as, Ytr_as, Xts_as, Yts_as = Xtr.to(device), Ytr.to(device).flatten(), Xts.to(device), Yts.to(device).flatten()
            act_block_size = block_size
            if act_block_size <= 0:
                act_block_size = Xtr.shape[0] // 100
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            kernel_params = get_kernel_params(Xtr, sigma)
            wrapper = ASkotchWrapper(
                Xtr_as, Ytr_as, Xts_as, Yts_as, act_block_size, precond_params, kernel_params, lam*Xtr_as.shape[0], task, num_iter, device
            )
            tr_time = time.time()
            wrapper.fit(Xtr_as, Ytr_as, Xts_as, Yts_as, err_fns[0])
            tr_time = time.time() - tr_time
            
            c_test_errs, c_train_errs, err_names, te_pred_time = test_model(wrapper, f"ASkotch on {dset}", Xts.to(device), Yts.to(device), Xtr.to(device), Ytr.to(device), err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            train_times.append(tr_time)
            test_pred_times.append(te_pred_time)
            torch.cuda.empty_cache()
            print(f"[--] Fold {it} -> test_err: {c_test_errs}\ttr_err: {c_train_errs}\terr_name: {err_names}\tte_pred_time: {te_pred_time}\ttr_time: {tr_time}")

        print_kfold_error_report(kfold, test_errs, train_errs, err_names, train_times, test_pred_times)
        print()


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
    device : str = 'cuda',
    seed : int = 124151
):
    sys.path.append(JOKER_BASE_PATH)
    from joker import InexactJoker, Joker
    from kernels.kernel import make_kernel
    from criterion import make_criterion

    seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = DataType.float32
#    if dtype.to_numpy_dtype() != np.float32:
#        raise RuntimeError(f"Joker can only run on single-precision floats.")

    err_fns = get_err_fns(dset)

    if kernel_type == "laplacian":
        joker_ktype = "lap"
    elif kernel_type == "gaussian":
        joker_ktype = "rbf"
    else:
        raise ValueError(f"Unrecognized kernel for Joker: {kernel_type}")
    crit = make_criterion(
        criterion, c=c, delta=delta_huber, eps=eps, dtype=dtype.to_torch_dtype()
    )

    def joker_get_kernel(training_data):
        if sigma < 0:
            sigma_square = get_median_sigma_square(training_data, num_samples=10_000)
            print(f"Using sigma = {float(np.sqrt(sigma_square)):.4f} from the median trick.")
        else:
            sigma_square = sigma ** 2
        if joker_ktype == "rbf":
            gamma = 0.5 / sigma_square
        elif joker_ktype == "lap":
            gamma = 1 / float(np.sqrt(sigma_square))
        else:
            raise RuntimeError(joker_ktype)
        return make_kernel(joker_ktype, gamma=gamma)
    

    def joker_init_model(train_x, train_y):
        kernel_func = joker_get_kernel(train_x)
        if inexact_type == 'rff':
            model = InexactJoker(
                train_x,
                train_y,
                dtype=dtype.to_torch_dtype(), 
                kernel=kernel_func, 
                criterion=crit,
                device=device,
                n_features=nrff,
                inexact_type=inexact_type,
                opt_blksz=block_size, #cfg["blksz"],
                incore=incore,#cfg["incore"],
                data_blksz=data_block_size,
                optim=opt_name,
            )
        elif inexact_type == 'fastfood':
            model = InexactJoker(
                train_x, 
                train_y, 
                dtype=dtype.to_torch_dtype(), 
                kernel=kernel_func, 
                criterion=crit,
                device=device,
                n_features=n_fastfood,
                inexact_type=inexact_type,
                opt_blksz=block_size, #cfg["blksz"],
                incore=incore,#cfg["incore"],
                data_blksz=data_block_size,
                optim=opt_name,
            )
        else:
            model = Joker(
                train_x, 
                train_y, 
                dtype=dtype.to_torch_dtype(), 
                kernel=kernel_func, 
                criterion=crit,
                device=device,
                opt_blksz=block_size, #cfg["blksz"],
                incore=incore,#cfg["incore"],
                data_blksz=data_block_size,
                optim=opt_name,
            )
        print(f"Starting to train Joker model {model} on data {dset} kernel {kernel_func}", flush=True)
        return model

    
    if kfold == 1:
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)

        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
        model = joker_init_model(Xtr, Ytr)
        
        tr_time = time.time()
        model.fit(
            max_iter=num_iter,
            max_iter_subprob=num_iter_subprob, #cfg["max_iter_subprob"],
            max_region_size=max_region_size, #cfg["max_trust_region_size"],
            region_shrink_freq=region_shrink_freq,
            verbose_freq=5000,
            region_shrink_rate=region_shrink_rate, #cfg["region_shrink_rate"],
            blk_strategy='random', #cfg["blk_strategy"],
            val_x=Xts,
            val_y=Yts,
            verbose_primal_dual=False
        )
        tr_time = time.time() - tr_time

        te_err, tr_err, err_names, te_pred_time = test_model(
            model, f"Joker on {dset}", Xts, Yts, Xtr, Ytr, err_fns
        )
        print(f"Joker timings. training={tr_time:.2f}s inference={te_pred_time:.2f}s")
    else:
        load_fn = get_cv_fn(dset)
        err_names = None
        test_errs, train_errs, train_times, test_pred_times = [], [], [], []
        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            print(f"Starting fold {it}")
            model = joker_init_model(Xtr, Ytr)
            tr_time = time.time()
            model.fit(
                max_iter=num_iter,
                max_iter_subprob=num_iter_subprob, #cfg["max_iter_subprob"],
                max_region_size=max_region_size, #cfg["max_trust_region_size"],
                region_shrink_freq=region_shrink_freq,
                region_shrink_rate=region_shrink_rate, #cfg["region_shrink_rate"],
                blk_strategy='random', #cfg["blk_strategy"],
                verbose_freq=5000,
                val_x=Xts,
                val_y=Yts,
                verbose_primal_dual=False
            )
            tr_time = time.time() - tr_time

            c_test_errs, c_train_errs, err_names, te_pred_time = test_model(
                model, f"Joker on {dset}", Xts, Yts, Xtr, Ytr, err_fns
            )
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            train_times.append(tr_time)
            test_pred_times.append(te_pred_time)
            torch.cuda.empty_cache()
            print(f"[--] Fold {it} -> test_err: {c_test_errs}\ttr_err: {c_train_errs}\terr_name: {err_names}\tte_pred_time: {te_pred_time}\ttr_time: {tr_time}")

        print_kfold_error_report(kfold, test_errs, train_errs, err_names, train_times, test_pred_times)
        print()
        with open(f"./joker_{criterion}_{inexact_type}_{dset}_{str(dtype)}_kfold_{kfold}.log", 'w') as f_out:
            f_out.write(f"[TIME] {np.mean(train_times)},{np.std(train_times)},{np.mean(test_pred_times)},{np.std(test_pred_times)}\n")
            f_out.write(f"[TEST PERFORMANCE]\n")
            for err_fn_i in range(len(err_fns)):
                mu_perf, std_perf = np.mean([e[err_fn_i] for e in test_errs]), np.std([e[err_fn_i] for e in test_errs])
                f_out.write(','.join([str(e[err_fn_i]) for e in test_errs]) +  f",{mu_perf},{std_perf}\n")
            f_out.write(f"[TRAIN PERFORMANCE]\n")
            for err_fn_i in range(len(err_fns)):
                mu_perf, std_perf = np.mean([e[err_fn_i] for e in train_errs]), np.std([e[err_fn_i] for e in train_errs])
                f_out.write(','.join([str(e[err_fn_i]) for e in train_errs]) +  f",{mu_perf},{std_perf}\n")
            f_out.flush()


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
    debug: bool,
):
    from falkon import kernels
    from falkon.models import falkon
    from falkon.utils import TicToc

    seed_all(seed)

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
        debug=debug,
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
            c_test_errs, c_train_errs, err_names, _ = test_model(flk, f"Falkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)

        print_kfold_error_report(kfold, test_errs, train_errs, err_fns)


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
            penalty=args.penalty,
            kernel=args.kernel,
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
            lam=args.penalty,
            kernel_type = args.kernel,
            rank=args.num_centers,
            num_iter=args.epochs,
            block_size=args.askotch_bs,
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
