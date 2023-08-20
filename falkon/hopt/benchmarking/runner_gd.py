import io
import pathlib
import subprocess
from typing import List, Union

SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("benchmark_cli.py").resolve()
DEFAULT_SEED = 123199
STOCH_MODELS = [
    "stoch-creg-penfit",
]
VAL_MODELS = [
    "holdout",
]
MODELS = [
    "sgpr",
    "gcv",
    "loocv",
    "holdout",
    "creg-notrace",
    "creg-penfit",
    "stoch-creg-penfit",
    "svgp",
]


def gen_exp_name(
    optimizer,
    num_centers,
    learning_rate,
    penalty,
    sigma,
    val_percentage,
    opt_m,
    sigma_type,
    model,
    dataset,
    num_trace_vecs,
    flk_maxiter,
    cg_tol,
    extra_name,
):
    val_pct_str = f"_val{val_percentage}_" if True else ""  # model in VAL_MODELS else ""
    trace_vec_str = f"_ste{num_trace_vecs}_" if model in STOCH_MODELS else ""
    flk_miter_str = f"_{flk_maxiter}fits_" if model in STOCH_MODELS else ""
    cg_tol_str = f"_cg{cg_tol:.1e}" if model in STOCH_MODELS else ""
    opt_m_str = "_optM_" if opt_m else ""

    return (
        f"{dataset}_hopt_{model}_test_hopt_{optimizer}_m{num_centers}_lr{learning_rate}_"
        f"pinit{penalty}_{sigma_type}sinit{sigma}{val_pct_str}{trace_vec_str}"
        f"{flk_miter_str}{cg_tol_str}{opt_m_str}_{extra_name}"
    )


def run_simple_hopt(
    sigma_init: Union[float, str],
    pen_init: Union[float, str],
    lr: float,
    num_epochs: int,
    M: int,
    dataset: str,
    val_pct: float,
    model: str,
    optim: str,
    sigma: str,
    opt_centers: bool,
    num_trace_vecs: int,
    flk_maxiter: int,
    exp_name: str,
    cg_tol: float,
    seed: int = DEFAULT_SEED,
):
    exp_name_final = gen_exp_name(
        optim,
        M,
        lr,
        pen_init,
        sigma_init,
        val_pct,
        opt_centers,
        sigma,
        model,
        dataset,
        num_trace_vecs,
        flk_maxiter,
        cg_tol,
        exp_name,
    )
    proc_args = [
        f"python {SIMPLE_HOPT_PATH}",
        f"--seed {seed}",
        f"--cg-tol {cg_tol}",
        f"--val-pct {val_pct}",
        f"--sigma-type {sigma}",
        f"--sigma-init {sigma_init}",
        f"--penalty-init {pen_init}",
        f"--lr {lr}",
        f"--epochs {num_epochs}",
        f"--optimizer {optim}",
        "--op",
        "--os",
        f"--num-centers {M}",
        f"--dataset {dataset}",
        f"--model {model}",
        f"--num-t {num_trace_vecs}",
        f"--flk-maxiter {flk_maxiter}",
        "--cuda",
        "--loss-every 2",
        "--early-stop-every 201",
        "--cgtol-decrease-every 10",
        f"--name {exp_name_final}",
    ]
    if model == "svgp":
        proc_args.append("--mb 16000")
    if opt_centers:
        proc_args.append("--oc")
    proc = subprocess.Popen([" ".join(proc_args)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open(f"logs/tee_{exp_name_final}.txt", "a+") as out_f:
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            print(line, end="")
            out_f.write(line)
    ret_code = proc.wait()
    if ret_code != 0:
        raise RuntimeError("Process returned error", ret_code)


def run_for_models(
    sigma_init: Union[float, str],
    pen_init: Union[float, str],
    lr: float,
    num_epochs: int,
    M: int,
    dataset: str,
    optim: str,
    val_pct: float,
    models: List[str],
    sigma: str,
    opt_centers: bool,
    num_trace_vecs: int,
    flk_maxiter: int,
    cg_tol: float,
    exp_name: str,
    num_rep: int = 1,
):
    for model in models:
        for i in range(num_rep):
            run_simple_hopt(
                sigma_init,
                pen_init,
                lr,
                num_epochs,
                M,
                dataset,
                val_pct,
                model,
                optim,
                seed=DEFAULT_SEED + i,
                sigma=sigma,
                opt_centers=opt_centers,
                num_trace_vecs=num_trace_vecs,
                flk_maxiter=flk_maxiter,
                exp_name=exp_name,
                cg_tol=cg_tol,
            )


def run_for_valpct(
    sigma_init: Union[float, str],
    pen_init: Union[float, str],
    lr: float,
    num_epochs: int,
    M: int,
    dataset: str,
    optim: str,
    val_pcts: List[float],
    sigma: str,
    opt_centers: bool,
    num_trace_vecs: int,
    flk_maxiter: int,
    cg_tol: float,
    exp_name: str,
    model: str = "hgrad-closed",
    num_rep: int = 1,
):
    for val_pct in val_pcts:
        for i in range(num_rep):
            run_simple_hopt(
                sigma_init,
                pen_init,
                lr,
                num_epochs,
                M,
                dataset,
                val_pct,
                model,
                optim,
                seed=DEFAULT_SEED + i,
                sigma=sigma,
                opt_centers=opt_centers,
                num_trace_vecs=num_trace_vecs,
                flk_maxiter=flk_maxiter,
                cg_tol=cg_tol,
                exp_name=exp_name,
            )


def run():
    datasets = [
        "protein",
        "chiet",
        "ictus",
        "codrna",
        "svmguide1",
        "phishing",
        "spacega",
        "cadata",
        "mg",
        "cpusmall",
        "abalone",
        "blogfeedback",
        "energy",
        "covtype",
        "ho-higgs",
        "ijcnn1",
        "road3d",
        "buzz",
        "houseelectric",
    ]
    datasets = ["svhn", "mnist-small", "fashionmnist", "cifar10"]
    datasets = [
        "protein",
        "chiet",
        "ictus",
        "codrna",
        "svmguide1",
        "phishing",
        "spacega",
        "cadata",
        "mg",
        "cpusmall",
        "abalone",
        "blogfeedback",
        "energy",
        "covtype",
        "ho-higgs",
        "ijcnn1",
        "road3d",
        "buzz",
        "houseelectric",
        "mnist-small",
        "svhn",
        "fashionmnist",
    ]
    datasets = ["flights"]
    num_epochs = 75
    learning_rate = 0.05
    M = 5000
    opt_m = True
    val_pct = 0.6
    optim = "adam"
    sigma = "single"
    extra_exp_name = "reb_fast-tr_v6"
    sigma_init = 1
    penalty_init = "auto"
    # Stochastic stuff
    flk_maxiter = 100
    num_trace_vecs = 20
    cg_tol = 1e-3
    # Models to use for training
    models = ["sgpr", "holdout", "gcv", "creg-notrace", "creg-penfit-special", "creg-penfit-divtr"]

    for dset in datasets:
        run_for_models(
            sigma_init=sigma_init,
            pen_init=penalty_init,
            lr=learning_rate,
            num_epochs=num_epochs,
            M=M,
            dataset=dset,
            val_pct=val_pct,
            models=models,
            num_rep=3,
            optim=optim,
            sigma=sigma,
            opt_centers=opt_m,
            exp_name=extra_exp_name,
            flk_maxiter=flk_maxiter,
            num_trace_vecs=num_trace_vecs,
            cg_tol=cg_tol,
        )


if __name__ == "__main__":
    run()
