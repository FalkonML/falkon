import io
import itertools
import pathlib
import subprocess

import numpy as np

SIMPLE_HOPT_PATH = pathlib.Path(__file__).parent.joinpath("benchmark_cli.py").resolve()
DEFAULT_SEED = 12319


def write_gridspec_file(out_file, sigmas, penalties):
    with open(out_file, "w") as fh:
        fh.write("sigma,penalty\n")
        for ex in itertools.product(sigmas, penalties):
            fh.write("%.8e,%.8e\n" % (ex[0], ex[1]))


def run_gs(
        val_pct: float,
        num_centers: int,
        dataset: str,
        model: str,
        gs_file: str,
        exp_name: str,
        seed: int = DEFAULT_SEED,):
    proc_args = [
        f"python {SIMPLE_HOPT_PATH}",
        f"--seed {seed}",
        "--cg-tol 1e-1",  # ignored
        f"--val-pct {val_pct}",
        "--sigma-type single",
        "--sigma-init 1.0",  # ignored
        "--penalty-init 1.0",  # ignored
        f"--num-centers {num_centers}",
        f"--dataset {dataset}",
        f"--model {model}",
        f"--grid-spec {gs_file}",
        "--cuda",
        f"--name {dataset}_gs_{model}_{exp_name}"
    ]
    if model == "svgp":
        proc_args.append("--mb 16000")
    proc = subprocess.Popen([" ".join(proc_args)], shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    for line in io.TextIOWrapper(proc.stdout, encoding='utf-8'):
        print(line, end='')
    ret_code = proc.wait()
    if ret_code != 0:
        raise RuntimeError("Process returned error", ret_code)


def run():
    gs_file = "tmp_gs_file"
    exp_name = "test_exp_m100_28_09"
    datasets = {
        "ho-higgs": {
            "num_centers": 100,
            "sigmas": np.logspace(-1, 2, 15),
            "penalties": np.logspace(-9, 0, 15),
            "train_size": 10000,
        },
        "svmguide1": {
            "num_centers": 100,
            "sigmas": np.logspace(-1, 1.5, 15),
            "penalties": np.logspace(-8, 0, 15),
            "train_size": 3089,
        },
        "boston": {
            "num_centers": 100,
            "sigmas": np.logspace(0, 1.5, 15),
            "penalties": np.logspace(-6, 2, 15),
            "train_size": 10,  # TODO: Fake but only useful for SVGP which is not tested
        },
        "energy": {
            "num_centers": 100,
            "sigmas": np.logspace(-1, 1.5, 15),
            "penalties": np.logspace(-8, 2, 15),
            "train_size": 614,
        },
        "protein": {
            "num_centers": 100,
            "sigmas": np.logspace(-1, 1.5, 15),
            "penalties": np.logspace(-8, 0, 15),
            "train_size": 36584,
        },
        "cpusmall": {
            "num_centers": 100,
            "sigmas": np.logspace(-1, 1.5, 15),
            "penalties": np.logspace(-9, 0, 15),
            "train_size": 5734,
        },
    }
    models = {
        "creg-notrace": {},
        "loocv": {},
        "gcv": {},
        "sgpr": {},
        "hgrad-closed": {'val_pct': 0.6},
        "creg-penfit": {},
    }
    for dset, dset_params in datasets.items():
        for model, model_params in models.items():
            penalties = dset_params['penalties']
            if model == 'svgp':
                min_penalty = 1e-4 / dset_params['train_size']
                penalties = np.logspace(np.log10(min_penalty), np.log10(dset_params['penalties'].max()), len(dset_params['penalties']))
            write_gridspec_file(gs_file, dset_params['sigmas'], penalties)
            run_gs(val_pct=model_params.get('val_pct', 0.2),
                   num_centers=dset_params['num_centers'],
                   dataset=dset,
                   model=model,
                   gs_file=gs_file,
                   exp_name=exp_name,
                   seed=DEFAULT_SEED)


if __name__ == "__main__":
    run()
