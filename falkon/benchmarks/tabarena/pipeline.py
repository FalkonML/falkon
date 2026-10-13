"""The BLOKS system pipeline: preprocessing, bagged inner CV, HPO and greedy ensembling.

Kept free of TabArena / AutoGluon imports so it can be used (and tested) on its own; the
``BloksSystemModel`` in ``system.py`` is a thin adapter around :class:`BloksPipeline`.

Per fit:

1. ``inner_folds``-fold CV on the training data (stratified for classification), or the inner splits
   passed to :meth:`BloksPipeline.fit` (the system passes TabArena's group / time-aware splits for tasks
   that declare them). The preprocessing is fit inside every inner fold, so nothing leaks into the
   validation rows.
2. Each HPO trial trains one BLOKS model per inner fold. Its out-of-fold (OOF) predictions are
   calibrated (classification: logistic calibration of the KRR scores, cross-fitted on the OOF
   scores) and scored with the task metric; that score drives the search (Optuna TPE / random).
   Trial 0 is always the fixed default configuration.
3. The ``pool_size`` best trials keep their fold models. A Caruana-style greedy ensemble with
   replacement is built over their OOF predictions (``ensemble_size`` steps; ``ensemble_size=1``
   keeps only the single best trial).
4. Prediction averages the fold models of every ensemble member (bagging, no refit), calibrates and
   mixes the members with the ensemble weights.

The time budget is respected between trials (a trial only starts if the recent trial time fits in
the remaining budget) and between the folds of a trial; the default trial always completes. If the
first fold of the default trial projects the trial past ``default_budget_fraction`` of the budget, the
number of Nystroem centers M is reduced (fit time is roughly linear in M) and the default trial restarts.

All trials use the same Nystroem centers per inner fold (the center-selection seed depends on the fold
only), which removes center-sampling noise from the HPO comparison and lets the kept models share them.
"""

from __future__ import annotations

import contextlib
import gc
import heapq
import io
import math
import time
from typing import Literal
import warnings
from collections.abc import Callable
import dataclasses
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, roc_auc_score


from falkon import kernels
import falkon
from falkon.models.bloks import Bloks
from falkon.options import FalkonOptions



optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------------------
# Chosen over the Gaussian / lambda=1e-6 start by a small check on sklearn's bundled datasets (diabetes,
# breast_cancer, wine, digits; 3 outer folds): lambda=1e-6 overfit clearly, Matern-1.5 at lambda=1e-4 had the
# best average rank. Revisit on TabArena-Lite.
DEFAULT_CONFIG = {
    "kernel": "matern1.5", 
    "sigma_mult": 1.0, 
    "penalty": 1e-4, 
    "num_transform": "quantile",
}
KERNELS = (
    "gaussian", 
    "laplacian", 
    "matern1.5", 
    "matern2.5"
)


def suggest_config(trial) -> dict:
    """The BLOKS search space (Optuna ``Trial`` API)."""
    return {
        "kernel": trial.suggest_categorical("kernel", list(KERNELS)),
        # Kernel length-scale as a multiple of the median pairwise distance of the (preprocessed) data.
        "sigma_mult": trial.suggest_float("sigma_mult", 0.05, 50.0, log=True),
        # Falkon-style penalty: the system solved is (Knm^T Knm + penalty * n * Kmm) a = Knm^T y.
        "penalty": trial.suggest_float("penalty", 1e-10, 1e-6, log=True),
        "num_transform": trial.suggest_categorical("num_transform", ["quantile", "standard"]),
        "pc_epsilon": trial.suggest_float("pc_epsilon", 1e-8, 1e-5, log=True),
    }


@dataclass
class BloksSolverSettings:
    """Solver settings that are fixed per system config (not tuned)."""
    max_centers: int = 20_000
    """Number of Nystroem centers M (capped at the inner-fold training size)."""
    block_size: int = 5_000
    """Preconditioner block size (M is rounded down to a multiple of it)."""
    maxiter: int = 30
    cg_tolerance: float = 5e-5
    dtype: str = "float32"
    recompute_blocks: bool = False
    """Recompute preconditioner blocks at every CG iteration (saves M * block_size memory)."""
    use_keops: bool = True
    store_kernel_d_threshold: int = 1500
    min_cuda_pc_size: int = 0
    """Smallest block that is factorized on GPU (falkon's own defaults are 10k / 30k)."""
    use_cpu: bool = False
    max_cpu_mem: float = float("inf")
    """Bytes of CPU RAM falkon may use (soft bound; decides e.g. whether K_nm is precomputed)."""
    min_centers: int = 1000
    """Lower bound when the budget guard reduces M."""
    debug: bool = False


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def _to_str(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype("object").where(df.notna(), "__nan__").astype(str)


class Preprocessor:
    """Preprocess data
    Numeric: median imputation + {quantile-normal, standardize-and-clip}, plus missing indicators.
    Categoricals: one-hot up to ``max_onehot`` levels, else (cross-fitted) target encoding.
    """

    def __init__(
        self, 
        problem_type: Literal["regression", "binary", "multiclass"], 
        num_transform: Literal["quantile", "standard"], 
        max_onehot: int, 
        seed: int
    ):
        self.problem_type = problem_type
        self.num_transform = num_transform
        self.max_onehot = max_onehot
        self.seed = seed

    def _datetimes_to_float(self, X: pd.DataFrame) -> pd.DataFrame:
        dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
        if not dt_cols:
            return X
        X = X.copy()
        for c in dt_cols:
            X[c] = (X[c].astype("int64") / 1e9).where(X[c].notna())
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        X = self._datetimes_to_float(X)
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if c not in num_cols]
        card = {c: X[c].nunique(dropna=False) for c in cat_cols}
        lo_cats = [c for c in cat_cols if 1 < card[c] <= self.max_onehot]
        hi_cats = [c for c in cat_cols if card[c] > self.max_onehot]

        if self.num_transform == "quantile":
            scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(2, min(1000, len(X))),
                subsample=200_000,
                random_state=self.seed,
            )
        elif self.num_transform == "standard":
            clip = FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0})
            scaler = Pipeline([("std", StandardScaler()), ("clip", clip)])
        else:
            raise ValueError(f"Unknown num_transform {self.num_transform!r}")

        transformers = []
        if len(num_cols) > 0:
            transformers.append(
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", scaler)]), num_cols)
            )
            if X[num_cols].isna().to_numpy().any():
                transformers.append(("nan", MissingIndicator(features="missing-only", error_on_new=False), num_cols))
        if len(lo_cats) > 0:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
            transformers.append(("ohe", Pipeline([("str", FunctionTransformer(_to_str)), ("ohe", ohe)]), lo_cats))
        if len(hi_cats) > 0:
            target_type = "continuous" if self.problem_type == "regression" else self.problem_type
            te = TargetEncoder(target_type=target_type, random_state=self.seed)
            transformers.append(
                (
                    "te",
                    Pipeline([("str", FunctionTransformer(_to_str)), ("te", te), ("sc", StandardScaler())]),
                    hi_cats,
                ),
            )
        if len(transformers) == 0:
            raise ValueError("No usable feature columns.")
        self.ct_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        out = self.ct_.fit_transform(X, y)
        return np.ascontiguousarray(out, dtype=np.float64)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.ascontiguousarray(
            self.ct_.transform(self._datetimes_to_float(X)), 
            dtype=np.float64
        )


def median_distance(X: np.ndarray, max_rows: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
    d = torch.pdist(torch.from_numpy(X[idx]))
    d = d[d > 0]
    return float(torch.median(d).item()) if d.numel() else 1.0


# --------------------------------------------------------------------------------------
# BLOKS model construction
# --------------------------------------------------------------------------------------
def _make_kernel(
    name: Literal["gaussian", "laplacian", "matern0.5", "matern1.5", "matern2.5", "materninf"], 
    sigma: float, 
    opt: FalkonOptions | None,
):
    if name == "gaussian":
        return kernels.GaussianKernel(sigma, opt=opt)
    if name == "laplacian":
        return kernels.LaplacianKernel(sigma, opt=opt)
    if name.startswith("matern"):
        return kernels.MaternKernel(sigma=sigma, nu=float(name[len("matern") :]), opt=opt)
    raise ValueError(f"Unknown kernel {name!r}")


def make_bloks(cfg: dict, sigma: float, n_train: int, s: BloksSolverSettings, seed: int):
    M = min(s.max_centers, n_train)
    bs = min(s.block_size, M)
    M = (M // bs) * bs  # Bloks requires block_size | M
    opt = falkon.FalkonOptions(
        compute_arch_speed=False,
        no_single_kernel=False,
        cg_tolerance=s.cg_tolerance,
        cg_stagnation_iterations=3,
        cg_stagnation_threshold=0.98,
        pc_epsilon_32=cfg["pc_epsilon"],
        pc_epsilon_64=cfg["pc_epsilon"],
        keops_active="force" if s.use_keops else "no",
        keops_sum_scheme="kahan_scheme",
        store_kernel_d_threshold=s.store_kernel_d_threshold,
        min_cuda_pc_size_32=s.min_cuda_pc_size,
        min_cuda_pc_size_64=s.min_cuda_pc_size,
        use_cpu=s.use_cpu,
        max_cpu_mem=s.max_cpu_mem,
        debug=s.debug,
    )
    return Bloks(
        kernel=_make_kernel(cfg["kernel"], sigma, opt),
        penalty=cfg["penalty"],
        M=M,
        block_size=bs,
        recompute_blocks=s.recompute_blocks,
        maxiter=s.maxiter,
        seed=seed,
        error_fn=None,
        options=opt,
    )


# --------------------------------------------------------------------------------------
# Targets, calibration, metrics
# --------------------------------------------------------------------------------------
def encode_targets(
    y: np.ndarray, 
    problem_type: Literal["regression", "binary", "multiclass"], 
    n_classes: int
) -> tuple[np.ndarray, None | tuple[float, float]]:
    """KRR targets for one fold, plus the (mean, std) needed to undo them (regression only)."""
    if problem_type == "regression":
        mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
        return ((y - mu) / sd)[:, None], (mu, sd)
    if problem_type == "binary":
        return (2.0 * y - 1.0)[:, None], None
    Y = -np.ones((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y, None


CALIBRATION_C = 1.0
"""Inverse L2 strength of the logistic calibrator. sklearn's objective is ``C * sum(loss) + ||w||^2 / 2``, so a
fixed C regularizes small datasets more than large ones. A nearly unregularized calibrator (C=1e4) turned
OOF-separable configs into overconfident test probabilities on small multiclass data."""


def _calibrator():
    return LogisticRegression(C=CALIBRATION_C, max_iter=5000)


def calibrated_proba(
    cal: LogisticRegression, 
    scores: np.ndarray, 
    n_classes: int
) -> np.ndarray:
    p = np.full((len(scores), n_classes), 1e-15)
    p[:, cal.classes_] = cal.predict_proba(scores)
    return p / p.sum(1, keepdims=True)


def crossfit_proba(scores: np.ndarray, y: np.ndarray, n_classes: int, seed: int) -> np.ndarray:
    """Out-of-sample calibrated probabilities for the OOF scores (5-fold cross-fitted calibrator)."""
    out = np.empty((len(y), n_classes))
    try:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(scores, y):
            out[te] = calibrated_proba(_calibrator().fit(scores[tr], y[tr]), scores[te], n_classes)
    except ValueError:  # very rare classes: fall back to an in-sample calibrator
        out = calibrated_proba(_calibrator().fit(scores, y), scores, n_classes)
    return out


def default_error_fn(problem_type: str, n_classes: int) -> Callable[[np.ndarray, np.ndarray], float]:
    """TabArena-v0.1's metrics (lower is better). ``pred`` is (n,) for regression, (n, K) probs otherwise."""
    if problem_type == "regression":
        return lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
    if problem_type == "binary":
        return lambda y, p: float(1.0 - roc_auc_score(y, p[:, 1]))
    return lambda y, p: float(log_loss(y, p, labels=np.arange(n_classes)))


class _TrialTimeoutError(Exception):
    pass


class _RescaleError(Exception):
    """The default trial would take too long with the current number of centers."""

    def __init__(self, M: int, projected_s: float):
        super().__init__(f"M={M}, projected {projected_s:.0f}s")
        self.M = M
        self.projected_s = projected_s


@dataclass
class _TrialResult:
    number: int
    params: dict
    val_error: float
    oof_pred: np.ndarray  # metric-space OOF predictions (regression values / calibrated probabilities)
    models: list  # per inner fold: (bloks_model, target_stats)
    calibrator: LogisticRegression | None  # classification: fit on all OOF scores; applied to test scores
    time_s: float


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------
class BloksPipeline:
    def __init__(
        self,
        *,
        problem_type: Literal["regression", "binary", "multiclass"],
        n_classes: int = 0,
        error_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        bloks_cfg: BloksSolverSettings | None = None,
        hpo: Literal["tpe", "random", "none"] = "tpe",
        n_trials: int = 200,
        n_startup_trials: int = 10,
        inner_folds: int = 8,
        ensemble_size: int = 25,
        pool_size: int = 20,
        max_onehot: int = 32,
        time_limit: float | None = None,
        time_safety: float = 0.9,
        default_budget_fraction: float | None = None,
        random_state: int = 0,
        log: Callable[[str], None] = print,
    ):
        """
        pool_size (int): number of candidate hyperparameters kept in pool for ensembling
        ensemble_size (int): number of models (with different hyperparameters) used for final
            ensembling

        """
        if hpo not in {"tpe", "random", "none"}:
            raise ValueError(f"hpo must be 'tpe', 'random' or 'none', got {hpo!r}")
        self.problem_type: Literal["regression", "binary", "multiclass"] = problem_type
        self.n_classes = n_classes
        self.error_fn = error_fn or default_error_fn(problem_type, n_classes)
        self.bloks_cfg = bloks_cfg or BloksSolverSettings()
        self.hpo = hpo
        self.n_trials = 1 if hpo == "none" else n_trials
        self.n_startup_trials = n_startup_trials
        self.inner_folds = inner_folds
        self.ensemble_size = max(1, ensemble_size)
        self.pool_size = 1 if self.ensemble_size == 1 else max(1, pool_size)
        self.max_onehot = max_onehot
        self.time_limit = time_limit
        self.time_safety = time_safety
        if default_budget_fraction is None:  # leave room for the search unless only the default runs
            default_budget_fraction = 0.8 if hpo == "none" else 0.3
        self.default_budget_fraction = default_budget_fraction
        self.random_state = random_state
        self.log = log

        self.centers_cache = {}
        self.preprocessed_cache = {}
        self.preprocessor_cache = {}

    def inner_splits(self, y: np.ndarray) -> list:
        k = max(2, min(self.inner_folds, len(y)))
        if self.problem_type == "regression":
            splitter = KFold(k, shuffle=True, random_state=self.random_state)
        else:
            k = max(2, min(k, int(np.bincount(y).max())))
            splitter = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
        return list(splitter.split(np.zeros(len(y)), y))

    def preprocess(
        self, 
        fold_id: int, 
        x_tr: pd.DataFrame,
        y_tr: np.ndarray,
        x_val: pd.DataFrame,
        num_transform: Literal["quantile", "standard"]
    ):
        """Fitted preprocessor + preprocessed train/val tensors of inner fold ``k`` (cached across trials)."""
        key = (fold_id, num_transform)
        if key not in self.preprocessed_cache:
            pp = Preprocessor(self.problem_type, num_transform, self.max_onehot, self.random_state + fold_id)
            x_tr_pp = pp.fit_transform(x_tr, y_tr)
            x_tr_pp_torch = torch.from_numpy(x_tr_pp).to(self.dtype)
            x_val_pp = pp.transform(x_val)
            x_val_pp_torch = torch.from_numpy(x_val_pp).to(self.dtype)
            self.preprocessor_cache[key] = (pp, median_distance(x_tr_pp, seed=self.random_state + fold_id))
            self.preprocessed_cache[key] = (x_tr_pp_torch, x_val_pp_torch)
        return (*self.preprocessed_cache[key], *self.preprocessor_cache[key])

    # ---- one configuration --------------------------------------------------------------
    def eval_one_config(
        self, 
        x_full: pd.DataFrame,
        y_full: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        trial: int, 
        cfg: dict, 
        deadline: float | None
    ) -> _TrialResult:
        t0 = time.monotonic()
        n_out = self.n_classes if self.problem_type == "multiclass" else 1
        oof = np.zeros((len(y_full), n_out))
        oof_count = np.zeros(len(y_full))
        models = []
        for fold_id, (tr, va) in enumerate(splits):
            t_fold = time.monotonic()
            if deadline is not None and trial > 0 and time.monotonic() > deadline:
                raise _TrialTimeoutError

            x_tr_pd = x_full.iloc[tr]
            x_val_pd = x_full.iloc[va]
            y_tr = y_full[tr]

            x_tr_final, x_val_final, preprocessor, med = self.preprocess(
                fold_id,
                x_tr_pd,
                y_tr,
                x_val_pd,
                cfg["num_transform"]
            )
            y_tr_enc, stats = encode_targets(y_tr, self.problem_type, self.n_classes)
            y_tr_final = torch.from_numpy(y_tr_enc).to(self.dtype)

            seed = self.random_state + fold_id  # same centers for every trial on this fold
            torch.manual_seed(seed)
            model = make_bloks(
                cfg, 
                cfg["sigma_mult"] * med, 
                x_tr_final.shape[0], 
                self.bloks_cfg, 
                seed=seed
            )

            # The balkon CG solver prints the residual at every iteration: silence it unless debugging.
            with contextlib.nullcontext() if self.bloks_cfg.debug else contextlib.redirect_stdout(io.StringIO()):
                model.fit(x_tr_final, y_tr_final)

            model.precond = None  # not needed for prediction (and can hold M x block_size)
            self.share_centers(model, fold_id, cfg["num_transform"])

            pva = model.predict(x_val_final).cpu().numpy()
            if not np.isfinite(pva).all():
                raise FloatingPointError("non-finite predictions")
            if stats is not None:
                pva = pva * stats[1] + stats[0]

            oof[va] += pva
            oof_count[va] += 1
            models.append((model, stats))
            if fold_id == 0 and trial == 0 and self._default_budget is not None:
                projected_time = (time.monotonic() - t_fold) * len(splits)
                assert model.M is not None
                if projected_time > self._default_budget and self.bloks_cfg.min_centers < model.M:
                    raise _RescaleError(model.M, projected_time)
        if (oof_count == 0).any():
            raise ValueError("The inner splits leave some training rows without a validation prediction.")
        oof /= oof_count[:, None]

        calibrator = None
        if self.problem_type == "regression":
            oof_pred = oof[:, 0]
        else:
            oof_pred = crossfit_proba(oof, y_full, max(self.n_classes, 2), self.random_state)
            calibrator = _calibrator().fit(oof, y_full)
        val_error = float(self.error_fn(y_full, oof_pred))
        if not math.isfinite(val_error):
            raise FloatingPointError("non-finite validation error")
        return _TrialResult(trial, cfg, val_error, oof_pred, models, calibrator, time.monotonic() - t0)

    def share_centers(self, model, fold_id: int, num_transform: str) -> None:
        """Point the model at an identical, already-kept center tensor 
        (saves M x d memory per model in the pool)."""
        key = (fold_id, num_transform, model.ny_points_.shape[0])
        shared = self.centers_cache.get(key)
        if shared is not None and torch.equal(shared, model.ny_points_):
            model.ny_points_ = shared
        else:
            self.centers_cache[key] = model.ny_points_

    def eval_and_rescale(
        self, 
        x_full: pd.DataFrame,
        y_full: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        cfg: dict, 
        deadline: float | None
    ) -> _TrialResult:
        """Trial 0, reducing M until its projected time fits ``default_budget_fraction`` of the budget."""
        while True:
            try:
                return self.eval_one_config(
                    x_full, y_full, splits, trial=0, cfg=cfg, deadline=deadline
                )
            except _RescaleError as e:
                new_M = int(e.M * self._default_budget / e.projected_s * 0.8)
                new_M = max(self.bloks_cfg.min_centers, min(new_M, e.M // 2 if new_M >= e.M else new_M))
                self.log(
                    f"[BLOKS] default trial projected at {e.projected_s:.0f}s with M={e.M} "
                    f"(allowed {self._default_budget:.0f}s): reducing M to {new_M}",
                )
                self.bloks_cfg = dataclasses.replace(self.bloks_cfg, max_centers=new_M)
                self.centers_cache.clear()

    @property
    def dtype(self):
        return torch.float64 if self.bloks_cfg.dtype == "float64" else torch.float32

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> BloksPipeline:
        """Fit on ``X``, ``y`` (labels encoded as 0..K-1 for classification).

        ``splits``: optional inner (train, validation) positional index pairs, possibly from several
        repeats; the OOF prediction of a row is averaged over the splits that validate it.
        """
        t_start = time.monotonic()
        deadline = None if self.time_limit is None else t_start + self.time_limit * self.time_safety

        X = X.reset_index(drop=True)
        y = np.asarray(y)

        if splits is not None:
            splits = [(np.asarray(a), np.asarray(b)) for a, b in splits]
        else:
            splits = self.inner_splits(y)

        self.preprocessed_cache, self.preprocessor_cache, self.centers_cache = {}, {}, {}

        self._default_budget = (
            None if self.time_limit is None else self.time_limit * self.time_safety * self.default_budget_fraction
        )

        if self.hpo == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.random_state, n_startup_trials=self.n_startup_trials)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        # First trial is the default configuration
        study.enqueue_trial(DEFAULT_CONFIG)

        pool: list[tuple[float, int, _TrialResult]] = []
        self.history_: list[dict] = []
        for trial_id in range(self.n_trials):
            if trial_id > 0 and deadline is not None:
                recent = [h["time_s"] for h in self.history_[-3:] if h["state"] == "complete"]
                if recent and time.monotonic() + float(np.median(recent)) > deadline:
                    self.log(f"[BLOKS] time budget reached after {trial_id} trials")
                    break

            trial = study.ask()
            cfg = suggest_config(trial)

            t_trial = time.monotonic()
            try:
                if trial_id == 0:
                    res = self.eval_and_rescale(
                        x_full=X, y_full=y, splits=splits, cfg=cfg, deadline=deadline
                    )
                else:
                    res = self.eval_one_config(
                        x_full=X, y_full=y, splits=splits, trial=trial_id, cfg=cfg, deadline=deadline
                    )
            except _TrialTimeoutError:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self.history_.append({"number": trial_id, "params": cfg, "state": "timeout"})
                self.log(f"[BLOKS] trial {trial_id} stopped: time budget reached")
                break
            except Exception as e:
                if type(e).__name__ == "OutOfMemoryError" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self.history_.append(
                    {
                        "number": trial_id,
                        "params": cfg,
                        "state": "failed",
                        "error": f"{type(e).__name__}: {e}",
                        "time_s": time.monotonic() - t_trial,
                    },
                )
                self.log(f"[BLOKS] trial {trial_id} failed: {type(e).__name__}: {str(e)}")
                continue
            study.tell(trial, res.val_error)
            self.history_.append(
                {
                    "number": trial_id,
                    "params": cfg,
                    "state": "complete",
                    "val_error": res.val_error,
                    "time_s": res.time_s,
                },
            )

            # Keep the fold models of the `pool_size` best trials only.
            heapq.heappush(pool, (-res.val_error, -trial_id, res))
            if len(pool) > self.pool_size:
                heapq.heappop(pool)
            best = -max(pool)[0]
            self.log(
                f"[BLOKS] trial {trial_id:3d} val_error={res.val_error:.5f} (best {best:.5f}) {res.time_s:.1f}s {cfg}",
            )
            gc.collect()

        # Free the cached training matrices; only the fitted preprocessors are needed from here on.
        self.preprocessed_cache.clear()
        del X
        gc.collect()
        if not pool:
            errors = [h.get("error") for h in self.history_]
            raise RuntimeError(f"All BLOKS trials failed: {errors[:3]}")

        members = sorted((r for _, _, r in pool), key=lambda r: r.val_error)
        self.weights_, self.val_error_ = self._greedy_ensemble(members, y)
        self.members_ = [m for m, w in zip(members, self.weights_, strict=True) if w > 0]
        self.weights_ = self.weights_[self.weights_ > 0]
        self.fit_time_s_ = time.monotonic() - t_start
        return self

    def _greedy_ensemble(self, members: list[_TrialResult], y: np.ndarray) -> tuple[np.ndarray, float]:
        """Caruana ensemble selection with replacement on the OOF predictions."""
        weights = np.zeros(len(members))
        if self.ensemble_size == 1 or len(members) == 1:
            weights[0] = 1.0
            return weights, members[0].val_error
        current, best_err = None, math.inf
        for step in range(self.ensemble_size):
            step_best = (math.inf, None, None)
            for j, m in enumerate(members):
                cand = m.oof_pred if current is None else (current * step + m.oof_pred) / (step + 1)
                err = float(self.error_fn(y, cand))
                if err < step_best[0]:
                    step_best = (err, j, cand)
            err, j, current = step_best
            weights[j] += 1
            best_err = err
        return weights / weights.sum(), best_err

    # ---- predict -------------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Regression: (n,) predictions. Classification: (n, n_classes) probabilities."""
        X = X.reset_index(drop=True)
        prepped = {}
        out = None
        for member, w in zip(self.members_, self.weights_, strict=True):
            scores = 0.0
            for k, (model, stats) in enumerate(member.models):
                key = (k, member.params["num_transform"])
                if key not in prepped:
                    prepped[key] = torch.from_numpy(self.preprocessor_cache[key][0].transform(X)).to(self.dtype)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = model.predict(prepped[key]).cpu().numpy()
                scores = scores + (p * stats[1] + stats[0] if stats is not None else p)
            scores = scores / len(member.models)
            if self.problem_type == "regression":
                pred = scores[:, 0]
            else:
                assert member.calibrator is not None
                pred = calibrated_proba(member.calibrator, scores, max(self.n_classes, 2))
            out = w * pred if out is None else out + w * pred
        return out

    def summary(self) -> dict:
        """JSON-serializable record of the fit, for the result metadata."""
        return {
            "val_error": float(self.val_error_),
            "fit_time_s": float(self.fit_time_s_),
            "n_trials": len(self.history_),
            "n_trials_complete": sum(h["state"] == "complete" for h in self.history_),
            "num_centers": int(self.members_[0].models[0][0].M),
            "ensemble": [
                {"trial": m.number, "weight": float(w), "val_error": float(m.val_error), "params": m.params}
                for m, w in zip(self.members_, self.weights_, strict=True)
            ],
            "history": self.history_,
            "solver": asdict(self.bloks_cfg),
        }