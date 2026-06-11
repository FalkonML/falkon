import argparse
import datetime
import functools

import numpy as np

from falkon.benchmarks.common.benchmark_utils import Dataset, DataType
from falkon.benchmarks.common.datasets import get_cv_fn, get_load_fn
from falkon.benchmarks.common.error_metrics import get_err_fns

import time




RANDOM_SEED = 123


def test_model(model, model_name, Xts, Yts, Xtr, Ytr, err_fns):
    te_pred_time = time.time()
    test_preds = model.predict(Xts)
    te_pred_time = time.time() - te_pred_time
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
    return test_errs, train_errs, te_pred_time


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
    import torch

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
        raise ValueError(f"Kernel {kernel} not understood for algorithm Balkon")

    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
        pc_epsilon_32=1e-6,
        pc_epsilon_64=1e-13,
        keops_active="force" if use_keops else "no",
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
        test_model(flk, f"Falkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
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
            c_test_errs, c_train_errs, _ = test_model(flk, f"Falkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
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



def run_askotch(
    dset: Dataset,
    data_path: str,
    dtype : DataType | None,
    task : str, # 'regression' or 'classification'
    kernel_type : str, # 'rbf' or 'matern'
#    precond_type : str, # 'nystrom
    sigma : float,  # used for both matern and rbf kernel
    nu : float, # used for matern kernel
    lam : float, # regularization
    rank : int,
    num_iter : int,
    block_size : int,
    kfold : int,
    device : str = 'cuda',
    seed : int = 124151
):
    import torch
    
    import pykeops
    print(pykeops.__version__)

    from pykeops.config import gpu_available
    print(gpu_available)

    from fast_krr.models import FullKRR
    from fast_krr.opts import ASkotchV2
    from tqdm import trange
    from falkon.benchmarks.models.askotch_model import ASkotchWrapper
    from falkon.utils import TicToc
    import time




    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = DataType.float32
    if dtype.to_numpy_dtype() != np.float32:
        raise RuntimeError(f"{algorithm} can only run on single-precision floats.")

    err_fns = get_err_fns(dset)

    if kernel_type == 'gaussian':
        kernel_params = {'type' : 'rbf', 'sigma' : sigma }
    
    precond_params = {"type": "nystrom", "r": rank, "rho": "damped"}
    
    

    
    if kfold == 1:
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)

        Xtr, Ytr, Xts, Yts = Xtr.to(device), Ytr.to(device).flatten(), Xts.to(device), Yts.to(device).flatten()

        if block_size <= 0:
            block_size = Xtr.shape[0] // 100


        # Small test
#        Xtr, Ytr = Xtr[:1000, :], Ytr[:1000]
#        Xts, Yts = Xts[:1000, :], Yts[:1000]
        

        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]


        # w0 = torch.zeros((Xtr.shape[0], ), device=device)
        # model = FullKRR(Xtr, Ytr, Xts, Yts, kernel_params=kernel_params, Ktr_needed=True, lambd=lam * Xtr.shape[0], task=task, w0=w0, device=device)

        # opt = ASkotchV2(model=model, block_sz=block_size, precond_params=precond_params)
        wrapper = ASkotchWrapper(Xtr, Ytr, Xts, Yts, block_size, precond_params, kernel_params, lam*Xtr.shape[0], task, num_iter, device)

#        with TicToc("ASkotch Algorithm"):
        tr_time = time.time()
        wrapper.fit(Xtr, Ytr, Xts, Yts, err_fns[0])
        tr_time = time.time() - tr_time

        print(f"[--] Train time: {tr_time}")                
        te_err, tr_err, te_pred_time = test_model(wrapper, f"ASkotch on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
        print(f"[--] Test errors: {te_err}")
        print(f"[--] Train errors: {tr_err}")
        with open(f"./askotch_{dset}_single_run.log", 'w') as f_out:
            f_out.write(','.join([str(e) for e in tr_err]) +"," + ','.join([str(e) for e in te_err]) +f",{tr_time},{te_pred_time}\n")
            f_out.flush()
    else:
        load_fn = get_cv_fn(dset)

        test_errs, train_errs, train_times, test_pred_times = [], [], [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            Xtr, Ytr, Xts, Yts = Xtr.to(device), Ytr.to(device).flatten(), Xts.to(device), Yts.to(device).flatten()
            if block_size <= 0:
                block_size = Xtr.shape[0] // 100
#            Small test
            # Xtr, Ytr = Xtr[:1000, :], Ytr[:1000]
            # Xts, Yts = Xts[:1000, :], Yts[:1000]

            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]
            # w0 = torch.zeros((Xtr.shape[0], ), device=device)
            # model = FullKRR(Xtr.to(device), Ytr.to(device).flatten(), Xts.to(device), Yts.to(device).flatten(), kernel_params=kernel_params, Ktr_needed=True, lambd=lam * Xtr.shape[0], task=task, w0=w0, device=device)

            # opt = ASkotchV2(model=model, block_sz=block_size, precond_params=precond_params)
            # wrapper = ASkotchWrapper(opt, num_iter)

            wrapper = ASkotchWrapper(Xtr, Ytr, Xts, Yts, block_size, precond_params, kernel_params, lam*Xtr.shape[0], task, num_iter, device)

            #with TicToc(f"ASkotch ALGORITHM (fold {it})"):
            tr_time = time.time()
            wrapper.fit(Xtr, Ytr, Xts, Yts, err_fns[0])
            tr_time = time.time() - tr_time
            
            c_test_errs, c_train_errs, te_pred_time = test_model(wrapper, f"ASkotch on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            train_times.append(tr_time)
            test_pred_times.append(te_pred_time)
            torch.cuda.empty_cache()

        print(f"Train time: {np.mean(train_times)} +/ {np.std(train_times)}")
        print(f"Test time: {np.mean(test_pred_times)} +/- {np.std(test_pred_times)}")
        print(f"Full errors: Test {test_errs} - Train {train_errs}")
        print()
        print(f"{kfold}-Fold Error Report")
        with open(f"./askotch_{dset}_kfold_{kfold}.log", 'w') as f_out:
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



def run_joker(
    dset: Dataset,
    data_path: str,
    dtype : DataType | None,
    num_iter_subprob : int,
    criterion : str,
    c : float, # penalty parameter of error term
    kernel_type : str, # 'rbf' or 'matern'
    sigma : float,  # used for both matern and rbf kernel
    lam : float, # regularization
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
    import torch
    import time
    import sys
 
    sys.path.append("./joker/src") 
    from joker import InexactJoker, Joker
    from kernels.kernel import make_kernel
    from criterion import make_criterion
    from optim import make_optimizer



    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = DataType.float32
    if dtype.to_numpy_dtype() != np.float32:
        raise RuntimeError(f"{algorithm} can only run on single-precision floats.")

    err_fns = get_err_fns(dset)

    
    
    
    if kfold == 1:
        load_fn = get_load_fn(dset)
        Xtr, Ytr, Xts, Yts, kwargs = load_fn(dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)

        Xtr, Ytr, Xts, Yts = Xtr.to(device), Ytr.to(device), Xts.to(device), Yts.to(device)

        n_sub = 10000
        sub_Xtrain = Xtr[:n_sub, :]
        sig2 = torch.mean(torch.pdist(sub_Xtrain) ** 2) # median trick        


        
        if kernel_type == 'gaussian':
            ktype = 'gaussian'
            gamma = 0.5 / (sigma**2) 
        elif kernel_type == 'lap':
            ktype = 'lap'
            gamma = 1/sigma 
        kernel = make_kernel(ktype, gamma=gamma) if sigma > 0 else make_kernel(ktype, gamma=1.0 / sig2, degree=2)

        err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]

        crit = make_criterion(criterion, c=c, delta=delta_huber, eps=eps, dtype=dtype.to_torch_dtype())
        #opt = make_optimizer(opt_name)
        
        if inexact_type == 'rff':
            model = InexactJoker(Xtr, Ytr, dtype=dtype.to_torch_dtype(), kernel=kernel, criterion=crit,
                    device=device,
                    n_features=nrff,
                    inexact_type=inexact_type,
                    opt_blksz=block_size, #cfg["blksz"],
                    incore=incore,#cfg["incore"],
                    data_blksz=data_block_size,
                    optim=opt_name)
        elif inexact_type == 'fastfood':
            model = InexactJoker(Xtr, Ytr, dtype=dtype.to_torch_dtype(), kernel=kernel, criterion=crit,
                    device=device,
                    n_features=n_fastfood,
                    inexact_type=inexact_type,
                    opt_blksz=block_size, #cfg["blksz"],
                    incore=incore,#cfg["incore"],
                    data_blksz=data_block_size,
                    optim=opt_name)
        else:
            model = Joker(Xtr, Ytr, dtype=dtype.to_torch_dtype(), kernel=kernel, criterion=crit,
                    device=device,
                    opt_blksz=block_size, #cfg["blksz"],
                    incore=incore,#cfg["incore"],
                    data_blksz=data_block_size,
                    optim=opt_name)
        
        tr_time = time.time()
        model.fit(max_iter=num_iter,
            max_iter_subprob=num_iter_subprob, #cfg["max_iter_subprob"],
            max_region_size=max_region_size, #cfg["max_trust_region_size"],
            region_shrink_freq=region_shrink_freq,
            verbose_freq=5000,
            region_shrink_rate=region_shrink_rate, #cfg["region_shrink_rate"],
            blk_strategy='random', #cfg["blk_strategy"],
            val_x=Xts,
            val_y=Yts,
            verbose_primal_dual=False)
        tr_time = time.time() - tr_time

        print(f"[--] Train time: {tr_time}")                
        te_err, tr_err, te_pred_time = test_model(model, f"Joker on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
        print(f"[--] Test errors: {te_err}")
        print(f"[--] Train errors: {tr_err}")
        with open(f"./joker_{dset}_single_run.log", 'w') as f_out:
            f_out.write(','.join([str(e) for e in tr_err]) +"," + ','.join([str(e) for e in te_err]) +f",{tr_time},{te_pred_time}\n")
            f_out.flush()
    else:
        load_fn = get_cv_fn(dset)

        test_errs, train_errs, train_times, test_pred_times = [], [], [], []

        for it, (Xtr, Ytr, Xts, Yts, kwargs) in enumerate(
            load_fn(k=kfold, dtype=dtype.to_numpy_dtype(), as_torch=True, path=data_path)
        ):
            Xtr, Ytr, Xts, Yts = Xtr.to(device), Ytr.to(device), Xts.to(device), Yts.to(device)

#            Small test
            # Xtr, Ytr = Xtr[:1000, :], Ytr[:1000]
            # Xts, Yts = Xts[:1000, :], Yts[:1000]


            n_sub = 10000
            sub_Xtrain = Xtr[:n_sub, :]
            sig2 = torch.mean(torch.pdist(sub_Xtrain) ** 2) # median trick        

            err_fns = [functools.partial(fn, **kwargs) for fn in err_fns]


            if kernel_type == 'gaussian':
                ktype = 'gaussian'
                gamma = 0.5 / (sigma**2) 
            elif kernel_type == 'lap':
                ktype = 'lap'
                gamma = 1/sigma 
            kernel = make_kernel(ktype, gamma=gamma) if sigma > 0 else make_kernel(ktype, gamma=1.0 / sig2, degree=2)


            crit = make_criterion(criterion, c=c, delta=delta_huber, eps=eps, dtype=dtype.to_torch_dtype())

            if inexact_type == 'rff':
                model = InexactJoker(Xtr, Ytr, dtype=dtype.to_torch_dtype(), kernel=kernel, criterion=crit,
                        device=device,
                        n_features=nrff,
                        inexact_type=inexact_type,
                        opt_blksz=block_size, #cfg["blksz"],
                        incore=incore,#cfg["incore"],
                        data_blksz=data_block_size,
                        optim=opt_name)
            elif inexact_type == 'fastfood':
                model = InexactJoker(Xtr, Ytr, dtype=dtype.to_torch_dtype(), kernel=kernel, criterion=crit,
                        device=device,
                        n_features=n_fastfood,
                        inexact_type=inexact_type,
                        opt_blksz=block_size, #cfg["blksz"],
                        incore=incore,#cfg["incore"],
                        data_blksz=data_block_size,
                        optim=opt_name)
            else:
                model = Joker(Xtr, Ytr, dtype=dtype.to_torch_dtype(), kernel=kernel, criterion=crit,
                        device=device,
                        opt_blksz=block_size, #cfg["blksz"],
                        incore=incore,#cfg["incore"],
                        data_blksz=data_block_size,
                        optim=opt_name)


            
            tr_time = time.time()
            model.fit(max_iter=num_iter,
                max_iter_subprob=num_iter_subprob, #cfg["max_iter_subprob"],
                max_region_size=max_region_size, #cfg["max_trust_region_size"],
                region_shrink_freq=region_shrink_freq,
                region_shrink_rate=region_shrink_rate, #cfg["region_shrink_rate"],
                blk_strategy='random', #cfg["blk_strategy"],
                verbose_freq=5000,
                val_x=Xts,
                val_y=Yts,
                verbose_primal_dual=False)
            tr_time = time.time() - tr_time

            
            c_test_errs, c_train_errs, te_pred_time = test_model(model, f"Joker on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
            train_errs.append(c_train_errs)
            test_errs.append(c_test_errs)
            train_times.append(tr_time)
            test_pred_times.append(te_pred_time)
            torch.cuda.empty_cache()

        print(f"Train time: {np.mean(train_times)} +/ {np.std(train_times)}")
        print(f"Pred test time: {np.mean(test_pred_times)} +/- {np.std(test_pred_times)}")
        print(f"Full errors: Test {test_errs} - Train {train_errs}")
        print()
        print(f"{kfold}-Fold Error Report")
        with open(f"./joker_{dset}_kfold_{kfold}.log", 'w') as f_out:
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
        raise ValueError(f"Kernel {kernel} not understood for algorithm Falkon")

    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=True,
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
            c_test_errs, c_train_errs, _ = test_model(flk, f"Falkon on {dset}", Xts, Yts, Xtr, Ytr, err_fns)
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


if __name__ == "__main__":
    print("-------------------------------------------")
    print(print(datetime.datetime.now()))
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument("-a", "--algorithm", type=str, choices=["falkon", "balkon", "askotch", "joker"])
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

    # Algo-specific
    p.add_argument("--balkon-block-size", type=int, required=False)

    ######### ASKOTCH PARAMS ##############
    p.add_argument('--askotch-task', default='classification', choices=['classification', 'regression'], help='Task tackled by ASkotch')
    p.add_argument('--askotch-bs', default=100, type=int, help='Block-size used in ASkotch')
    p.add_argument('--nu', default=5/2, type=float, help='nu of Matern kernel')
    
    ########## JOKER PARAMS
    p.add_argument('--num-iter-subprob', default=50, type=int, help='Number of inner iteration of Joker')
    p.add_argument('--joker-criterion', type=str, default="mse", choices=["mse", "huber", "svm", "log", "svr"], help="Which criterion to use for Joker")
    p.add_argument('--joker-c', type=float, default=1.0, help='Penalty parameter C of the error term in Joker')
    p.add_argument('--joker-delta-huber', type=float, default=-1, help='Joker\'s parameter delta')
    p.add_argument('--joker-eps', type=float, default=-1, help='Joker\'s parameter epsilon')
    p.add_argument("--joker-data-blksz", type=int, default=2048, help="The block size for data loading")
    p.add_argument("--joker-blksz", type=int, default=512, help="The block size for optimization")
    #p.add_argument("--blk_strategy", type=str, default="random", choices=["random", "cyclic"], help="How to choose the block during optimization")
#    p.add_argument('--max_iter', type=int, default=20000, help='Maximum number of iterations for fitting')
    p.add_argument('--max_iter_subprob', type=int, default=50, help='Maximum number of iterations for fitting')
    p.add_argument("--joker-max-trust-region-size", type=float, default=64, help="The maximal trust region size")
    p.add_argument("--region_shrink_freq", type=int, default=1000, help="Frequency of region shrink")
    p.add_argument("--region_shrink_rate", type=float, default=0.5, help="Rate of region shrink")
#   
#    p.add_argument('--joker-kernel', type=str, default="rbf", choices=["rbf", "student", "lap"], help="Kernel to use")
    p.add_argument("--joker-opt-name", type=str, default="trust_region", choices=["trust_region", "tncg"], help="Optimizer for block subproblems")
    p.add_argument('--joker-incore', action='store_true', help='Set this to place the data into GPU (Joker)')
#.add_argument("--n_rff", type=int, default=50000, help="Number of samples for RFF approximation, negative means full kernel")
    p.add_argument('--joker-inexact-type', type=str, default='rff', choices=['fastfood', 'rff', 'no'], help='Type of inexactness (Joker)')
    p.add_argument('--joker-nrff', type=int, default=50000, help="Number of samples for RFF approximation")
    p.add_argument("--joker-n-fastfood", type=int, default=100, help="Number of samples for Fastfood approximation")

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
            nu=args.nu,
            block_size=args.askotch_bs,
            kfold=args.kfold,
            seed=args.seed
        )
    elif args.algorithm == "joker":
    # dset: Dataset,
    # data_path: str,
    # dtype : DataType | None,
    # num_iter_subprob : int,
    # criterion : str,
    # c : float, # penalty parameter of error term
    # kernel_type : str, # 'rbf' or 'matern'
    # sigma : float,  # used for both matern and rbf kernel
    # lam : float, # regularization
    # num_iter : int,
    # block_size : int,
    # data_block_size : int,
    # max_region_size : float,
    # opt_name : str, #trust_region or tncg
    # kfold : int,
    # incore : bool,
    # region_shrink_freq : int = 1000,
    # region_shrink_rate : float = 0.5,
    # delta_huber : float = -1.0, 
    # eps : float = -1.0,
    # device : str = 'cuda',
    # seed : int = 124151

        run_joker(
            dset = args.dataset,
            data_path=args.data_path,
            dtype=args.dtype,
            num_iter_subprob=args.num_iter_subprob,
            criterion=args.joker_criterion,
            c = args.joker_c,
            data_block_size = args.joker_data_blksz,
            block_size=args.joker_blksz,
            sigma=args.sigma,
            lam=args.penalty,
            max_region_size = args.joker_max_trust_region_size,
            opt_name = args.joker_opt_name,
            kernel_type = args.kernel,
            region_shrink_freq = args.region_shrink_freq,
            region_shrink_rate = args.region_shrink_rate,
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
