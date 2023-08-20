import argparse
import datetime
import math
import os
import pickle
import warnings
from functools import partial
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from kernels import GaussianKernel

from falkon.benchmarks.common.benchmark_utils import Dataset
from falkon.benchmarks.common.datasets import get_load_fn
from falkon.benchmarks.common.error_metrics import get_err_fns
from falkon.center_selection import UniformSelector
from falkon.hopt.optimization.gd_train import train_complexity_reg, train_complexity_reg_mb
from falkon.hopt.optimization.grid_search import HPGridPoint, run_on_grid
from falkon.hopt.optimization.models import init_model
from falkon.hopt.optimization.reporting import get_writer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
AUTO_PEN_MULTIPLIER = 1


def median_heuristic(X: torch.Tensor, sigma_type: str, num_rnd_points: Optional[int]):
    # https://arxiv.org/pdf/1707.07269.pdf
    if num_rnd_points is not None and num_rnd_points < X.shape[0]:
        rnd_idx = np.random.choice(X.shape[0], size=num_rnd_points, replace=False)
        X = X[rnd_idx]
    if sigma_type == "diag":
        sigmas = [median_heuristic(X[:, i : i + 1], "single", None) for i in range(X.shape[1])]
        return torch.tensor(sigmas)
    else:
        # Calculate pairwise distances
        dist = torch.pdist(X, p=2)
        med_dist = torch.median(dist)
        return med_dist


def save_logs(logs: Any, exp_name: str, log_folder: str = "./logs"):
    new_exp_name = exp_name
    log_path = os.path.join(log_folder, f"{new_exp_name}.pkl")
    name_counter = 1
    while os.path.isfile(log_path):
        name_counter += 1
        new_exp_name = f"{exp_name}_{name_counter}"
        log_path = os.path.join(log_folder, f"{new_exp_name}.pkl")
    if new_exp_name != exp_name:
        warnings.warn(f"Logs will be saved to '{log_path}' because original name was taken.")

    with open(log_path, "wb") as file:
        pickle.dump(logs, file, protocol=4)
    print(f"Log saved to {log_path}", flush=True)


def read_gs_file(file_name: str) -> List[HPGridPoint]:
    df = pd.read_csv(file_name, header=0, index_col=False)
    points = []
    for row in df.itertuples():
        point = HPGridPoint(attributes=row._asdict())
        points.append(point)
    return points


def sigma_pen_init(
    data, sigma_type: str, sigma_init: Union[float, str], penalty_init: Union[float, str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sigma_init == "auto":
        sigma_init = median_heuristic(data["Xtr"], sigma_type="single", num_rnd_points=5000)
        print("Initial sigma is: %.4e" % (sigma_init))
    else:
        sigma_init = float(sigma_init)
    if sigma_type == "single":
        sigma_init = torch.tensor([sigma_init])
    elif sigma_type == "diag":
        d = data["Xtr"].shape[1]
        sigma_init = torch.tensor([sigma_init] * d)
    else:
        raise ValueError("sigma_type %s unrecognized" % (sigma_type))

    if penalty_init == "auto":
        penalty_init = AUTO_PEN_MULTIPLIER / data["Xtr"].shape[0]
        print(f"Initial penalty (auto with multiplier {AUTO_PEN_MULTIPLIER}) is: {penalty_init:.4e})")
    else:
        penalty_init = float(penalty_init)

    return sigma_init, penalty_init


def choose_centers_init(Xtr, metadata, num_centers, seed) -> torch.Tensor:
    if "centers" in metadata:
        print(f"Ignoring default centers and picking new {num_centers} centers.")
    selector = UniformSelector(np.random.default_rng(seed), num_centers)
    centers = selector.select(Xtr, None)
    # noinspection PyTypeChecker
    return centers


def run_grid_search(
    exp_name: str,
    dataset: Dataset,
    model_type: str,
    penalty_init: str,
    sigma_type: str,
    sigma_init: str,
    gs_file: str,
    num_centers: int,
    val_pct: float,
    per_iter_split: bool,
    cg_tol: float,
    flk_maxiter: int,
    num_trace_vecs: int,
    minibatch: int,
    cuda: bool,
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float64, as_torch=True)
    err_fns = get_err_fns(dataset)

    centers_init = choose_centers_init(Xtr, metadata, num_centers, seed)
    sigma_init, penalty_init = sigma_pen_init(
        {"Xtr": Xtr}, sigma_type=sigma_type, sigma_init=sigma_init, penalty_init=penalty_init
    )
    kernel = GaussianKernel(sigma=sigma_init.requires_grad_(False))
    grid_spec = read_gs_file(gs_file)
    model = init_model(
        model_type=model_type,
        data={"X": Xtr, "Y": Ytr},
        kernel=kernel,
        penalty_init=penalty_init,
        centers_init=centers_init,
        opt_penalty=True,  # Only to avoid errors, opt is not performed
        opt_centers=True,
        cuda=cuda,
        val_pct=val_pct,
        per_iter_split=per_iter_split,
        cg_tol=cg_tol,
        num_trace_vecs=num_trace_vecs,
        flk_maxiter=flk_maxiter,
    )
    logs = run_on_grid(
        Xtr=Xtr,
        Ytr=Ytr,
        Xts=Xts,
        Yts=Yts,
        model=model,
        err_fn=partial(err_fns[0], **metadata),
        grid_spec=grid_spec,
        cuda=cuda,
        minibatch=minibatch,
    )
    save_logs(logs, exp_name=exp_name)


def run_optimization(
    exp_name: str,
    dataset: Dataset,
    model_type: str,
    penalty_init: str,
    sigma_type: str,
    sigma_init: str,
    opt_centers: bool,
    opt_sigma: bool,
    opt_penalty: bool,
    num_centers: int,
    num_epochs: int,
    learning_rate: float,
    val_pct: float,
    per_iter_split: bool,
    cg_tol: float,
    flk_maxiter: int,
    num_trace_vecs: int,
    optimizer: str,
    cuda: bool,
    seed: int,
    minibatch: int,
    loss_every: int,
    early_stop_epochs: int,
    cgtol_decrease_epochs: Optional[int],
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xts, Yts, metadata = get_load_fn(dataset)(np.float32, as_torch=True)
    err_fns = get_err_fns(dataset)

    sigma_init, penalty_init = sigma_pen_init(
        {"Xtr": Xtr}, sigma_type=sigma_type, sigma_init=sigma_init, penalty_init=penalty_init
    )
    kernel = GaussianKernel(sigma=sigma_init.requires_grad_(opt_sigma))
    centers_init = choose_centers_init(Xtr, metadata, num_centers, seed)
    if minibatch > 0:
        num_batches = math.ceil(Xtr.shape[0] / minibatch)
        learning_rate = learning_rate / num_batches
        print("Learning rate changed to %.2e" % (learning_rate))

    model = init_model(
        model_type=model_type,
        data={"X": Xtr, "Y": Ytr},
        kernel=kernel,
        penalty_init=penalty_init,
        centers_init=centers_init,
        opt_penalty=opt_penalty,
        opt_centers=opt_centers,
        cuda=cuda,
        val_pct=val_pct,
        per_iter_split=per_iter_split,
        cg_tol=cg_tol,
        num_trace_vecs=num_trace_vecs,
        flk_maxiter=flk_maxiter,
    )
    if minibatch <= 0:
        logs = train_complexity_reg(
            Xtr=Xtr,
            Ytr=Ytr,
            Xts=Xts,
            Yts=Yts,
            model=model,
            err_fn=partial(err_fns[0], **metadata),
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cuda=cuda,
            loss_every=loss_every,
            optimizer=optimizer,
            early_stop_epochs=early_stop_epochs,
            cgtol_decrease_epochs=cgtol_decrease_epochs,
        )
    else:
        logs = train_complexity_reg_mb(
            Xtr=Xtr,
            Ytr=Ytr,
            Xts=Xts,
            Yts=Yts,
            model=model,
            err_fn=partial(err_fns[0], **metadata),
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cuda=cuda,
            loss_every=loss_every,
            optimizer=optimizer,
            minibatch=minibatch,
            early_stop_epochs=early_stop_epochs,
            cgtol_decrease_epochs=cgtol_decrease_epochs,
        )
    save_logs(logs, exp_name=exp_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FALKON Benchmark Runner")

    p.add_argument("-n", "--name", type=str, required=True)
    p.add_argument("-d", "--dataset", type=Dataset, choices=list(Dataset), required=True, help="Dataset")
    p.add_argument("-s", "--seed", type=int, required=True, help="Random seed")
    p.add_argument("--model", type=str, required=True, help="Model type")
    p.add_argument(
        "--cg-tol", type=float, required=False, default=1e-5, help="CG algorithm tolerance for hyper-gradient models."
    )
    p.add_argument("--lr", type=float, help="Learning rate for the outer-problem solver", default=0.01)
    p.add_argument("--epochs", type=int, help="Number of outer-problem steps", default=100)
    p.add_argument("--sigma-type", type=str, help="Use diagonal or single lengthscale for the kernel", default="single")
    p.add_argument("--sigma-init", type=str, default="2.0", help="Starting value for sigma")
    p.add_argument("--penalty-init", type=str, default="1.0", help="Starting value for penalty")
    p.add_argument("--oc", action="store_true", help="Whether to optimize Nystrom centers")
    p.add_argument("--os", action="store_true", help="Whether to optimize kernel parameters (sigma)")
    p.add_argument("--op", action="store_true", help="Whether to optimize penalty")
    p.add_argument("--num-centers", type=int, default=1000, required=False, help="Number of Nystrom centers for Falkon")
    p.add_argument("--val-pct", type=float, default=0, help="Fraction of validation data (hgrad experiments)")
    p.add_argument(
        "--per-iter-split",
        action="store_true",
        help="For validation-split based objectives, determines whether the split is "
        "changed at each iteration or not.",
    )
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument(
        "--grid-spec",
        type=str,
        default=None,
        help="Grid-spec file. Triggers a grid-search run instead of optimization.",
    )
    p.add_argument("--num-t", type=int, default=20, help="Number of trace-vectors for STE")
    p.add_argument(
        "--flk-maxiter", type=int, default=20, help="Maximum number of falkon iterations (for stochastic estimators)"
    )
    p.add_argument("--mb", type=int, default=0, required=False, help="mini-batch size. If <= 0 will use full-gradient")
    p.add_argument(
        "--loss-every",
        type=int,
        default=1,
        help="How often to compute the full loss (usually with retraining) in terms " "of training epochs.",
    )
    p.add_argument(
        "--early-stop-every",
        type=int,
        default=201,
        help="How many epochs without training loss improvements before stopping the " "optimization.",
    )
    p.add_argument(
        "--cgtol_decrease_every",
        type=int,
        default=999,
        help="Every how many epochs to decrease the convergence tolerance.",
    )
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fetch-loss", action="store_true")
    args = p.parse_args()
    print("-------------------------------------------")
    print(datetime.datetime.now())
    print("############### SEED: %d ################" % (args.seed))
    print("-------------------------------------------")
    np.random.seed(args.seed)
    get_writer(args.name)
    args.model = args.model.lower()

    if torch.cuda.is_available():
        torch.cuda.init()

    if args.grid_spec is not None:
        run_grid_search(
            exp_name=args.name,
            dataset=args.dataset,
            model_type=args.model,
            penalty_init=args.penalty_init,
            sigma_type=args.sigma_type,
            gs_file=args.grid_spec,
            sigma_init=args.sigma_init,
            num_centers=args.num_centers,
            val_pct=args.val_pct,
            per_iter_split=args.per_iter_split,
            cg_tol=args.cg_tol,
            cuda=args.cuda,
            seed=args.seed,
            num_trace_vecs=args.num_t,
            flk_maxiter=args.flk_maxiter,
            minibatch=args.mb,
        )
    else:
        run_optimization(
            exp_name=args.name,
            dataset=args.dataset,
            model_type=args.model,
            penalty_init=args.penalty_init,
            sigma_type=args.sigma_type,
            sigma_init=args.sigma_init,
            opt_centers=args.oc,
            opt_sigma=args.os,
            opt_penalty=args.op,
            num_centers=args.num_centers,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            val_pct=args.val_pct,
            per_iter_split=args.per_iter_split,
            cg_tol=args.cg_tol,
            cuda=args.cuda,
            seed=args.seed,
            optimizer=args.optimizer,
            num_trace_vecs=args.num_t,
            flk_maxiter=args.flk_maxiter,
            minibatch=args.mb,
            loss_every=args.loss_every,
            early_stop_epochs=args.early_stop_every,
            cgtol_decrease_epochs=args.cgtol_decrease_every,
        )
