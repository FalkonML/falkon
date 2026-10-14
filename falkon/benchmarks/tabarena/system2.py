from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, ClassVar
import dataclasses

import numpy as np
import pandas as pd
import torch

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.benchmark.exec_models.autogluon_utils import resolve_validation_splits
from tabarena.benchmark.validation_protocol import ValidationProtocol, ValidationResolution, structure_of

from falkon.benchmarks.tabarena.pipeline import BloksPipeline, BloksSolverSettings

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer
    from tabarena.benchmark.task.metadata import ValidationMetadata


def _log(msg: str) -> None:
    # flush=True: SLURM block-buffers stdout, so unflushed progress lines are invisible until the end.
    print(f"[BLOKS {time.strftime('%H:%M:%S')}] {msg.removeprefix('[BLOKS] ')}", flush=True)


def make_error_fn(eval_metric: Scorer, problem_type: str):
    """Wrap the task's AutoGluon ``Scorer`` as ``error(y_true_encoded, pred) -> float`` (lower is better).

    ``pred`` is the pipeline's output: (n,) values for regression and (n, n_classes) probabilities
    for classification. It is converted to the input the scorer expects (class labels, positive-class
    probability for binary scores, or the probability matrix).
    """

    def error(y_true: np.ndarray, pred: np.ndarray) -> float:
        if problem_type == "regression":
            return float(eval_metric.error(y_true, pred))
        if eval_metric.needs_pred or eval_metric.needs_class:
            return float(eval_metric.error(y_true, pred.argmax(axis=1)))
        if problem_type == "binary":
            return float(eval_metric.error(y_true, pred[:, 1]))
        return float(eval_metric.error(y_true, pred))

    return error


@contextlib.contextmanager
def _torch_threads(num_threads: int | None):
    """Cap torch's intra-op threads for the wrapped call and restore the previous value afterwards."""
    previous = torch.get_num_threads()
    if num_threads:
        torch.set_num_threads(int(num_threads))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def get_inner_splits(validation_metadata: ValidationMetadata, X: pd.DataFrame, y: pd.Series, inner_folds: int):
    """Inner splits for a task with a group / time structure, via TabArena's own resolver.

    Returns ``(splits, resolution)``: ``splits`` is ``None`` for an IID task (the pipeline then uses
    plain (stratified) k-fold); ``resolution`` is the ``ValidationResolution`` for the result record.
    For a structured task this is the same machinery (and clamps) the bagged TabArena models use.
    """
    structured = validation_metadata.group_on is not None or validation_metadata.time_on is not None
    if not structured:
        resolution = ValidationResolution(
            regime="explicit",
            num_bag_folds_nominal=inner_folds,
            num_bag_sets_nominal=1,
            structure=structure_of(validation_metadata),
        )
        return None, resolution
    protocol = ValidationProtocol(
        name="BLOKS",
        num_bag_folds=inner_folds,
        num_bag_sets=1,
        task_specific_validation=True,
    )
    clamps: list[str] = []
    splits, num_folds, num_repeats = resolve_validation_splits(
        validation_metadata,
        protocol,
        X=X.reset_index(drop=True),  # the resolver returns index labels: make them positions
        y=y.reset_index(drop=True),
        clamps=clamps,
    )
    resolution = ValidationResolution(
        regime="explicit",
        num_bag_folds_nominal=inner_folds,
        num_bag_sets_nominal=1,
        num_bag_folds_resolved=num_folds,
        num_bag_sets_resolved=num_repeats,
        clamps=tuple(clamps),
        custom_splits=splits is not None,
        num_custom_splits=len(splits) if splits is not None else None,
        structure=structure_of(validation_metadata),
    )
    return splits, resolution


class BloksSystemModel(ExternalSystemModel):
    """BLOKS (block-preconditioned Nystroem kernel ridge regression, falkon ``balkon`` branch) as a system.

    The system owns its whole pipeline (see ``_pipeline.py``): per-fold preprocessing, 8-fold bagged
    inner CV, Optuna HPO over kernel / length-scale / penalty / numeric scaling within the time budget,
    logistic calibration for classification, and a greedy (Caruana) ensemble over the best trials.

    Init hyperparameters (each a per-config knob for the system generator):

    * ``hpo`` — ``"tpe"`` (default), ``"random"`` or ``"none"`` (only the fixed default config).
    * ``n_trials`` — cap on HPO trials; the fit's ``time_limit`` usually binds first.
    * ``ensemble_size`` — greedy-ensemble steps over the ``pool_size`` best trials; ``1`` = best trial only.
    * ``inner_folds`` — bagging folds. Tasks with a ``group_on`` / ``time_on`` structure get group / time
      aware folds from TabArena's resolver (``resolve_validation_splits``), with its clamps.
    * ``default_budget_fraction`` — share of the budget the default trial may take before M is reduced
      (``None``: 0.3 when tuning, 0.8 for ``hpo="none"``).
    * ``max_centers``, ``block_size``, ``maxiter``, ``dtype``, ... — BLOKS solver settings
      (see ``BloksSolverSettings``).

    The compute and time budgets come per split from the runner: ``num_gpus == 0`` forces CPU,
    ``num_cpus`` caps torch's threads (restored after fit / predict), ``memory_limit`` becomes falkon's
    ``max_cpu_mem`` (and switches to recomputed preconditioner blocks when keeping them would be large),
    and ``time_limit`` bounds the HPO loop.

    Codebase: https://github.com/FalkonML/falkon/tree/balkon
    """

    # Import falkon / optuna and create the CUDA context outside the timed fit.
    warmup_modules: ClassVar[tuple[str, ...]] = ("falkon", "optuna")
    warmup_torch_device: ClassVar[bool] = True
    # BLOKS reports its OOF (bagged) validation error.
    can_get_error_val = True

    def __init__(
        self,
        *,
        hpo: str = "tpe",
        n_trials: int = 200,
        n_startup_trials: int = 10,
        ensemble_size: int = 25,
        pool_size: int = 20,
        inner_folds: int = 8,
        max_onehot: int = 32,
        time_safety: float = 0.9,
        default_budget_fraction: float | None = None,
        max_centers: int = 20_000,
        block_size: int = 5_000,
        maxiter: int = 30,
        cg_tolerance: float = 5e-5,
        dtype: str = "float64",
        recompute_blocks: bool = False,
        use_keops: bool = False,
        min_cuda_pc_size: int = 0,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hpo = hpo
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.ensemble_size = ensemble_size
        self.pool_size = pool_size
        self.inner_folds = inner_folds
        self.max_onehot = max_onehot
        self.time_safety = time_safety
        self.default_budget_fraction = default_budget_fraction
        self.bloks_cfg = BloksSolverSettings(
            max_centers=max_centers,
            block_size=block_size,
            maxiter=maxiter,
            cg_tolerance=cg_tolerance,
            dtype=dtype,
            recompute_blocks=recompute_blocks,
            use_keops=use_keops,
            min_cuda_pc_size=min_cuda_pc_size,
            debug=debug,
        )
        self._pipeline = None
        self._classes = None
        self._fit_resources: dict = {}
        self._validation_resolution = None
        self._num_cpus: int | None = None

    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        return False  # no code path of this system starts Ray

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        use_cpu = num_gpus == 0 or not torch.cuda.is_available()
        self._fit_resources = {
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "memory_limit": memory_limit,
            "device": "cpu" if use_cpu else "cuda",
        }

        # Labels: the system receives the raw labels (``preprocess_label = False``) and encodes them itself.
        if problem_type == "regression":
            y_enc, n_classes = y.to_numpy(dtype=np.float64), 0
        else:
            self._classes = pd.Index(pd.unique(y.dropna())).sort_values()
            y_enc = self._classes.get_indexer(y).astype(np.int64)
            n_classes = len(self._classes)

        splits, self._validation_resolution = get_inner_splits(validation_metadata, X, y, self.inner_folds)

        bloks_cfg = dataclasses.replace(self.bloks_cfg, use_cpu=use_cpu)
        if memory_limit:
            mem_bytes = float(memory_limit) * 2**30
            bloks_cfg = dataclasses.replace(bloks_cfg, max_cpu_mem=0.5 * mem_bytes)
            M = min(bloks_cfg.max_centers, len(X))
            itemsize = 8 if bloks_cfg.dtype == "float64" else 4
            if not bloks_cfg.recompute_blocks and M * min(bloks_cfg.block_size, M) * itemsize > 0.25 * mem_bytes:
                _log("in-memory preconditioner blocks would exceed 25% of memory_limit: recomputing them instead")
                bloks_cfg = dataclasses.replace(bloks_cfg, recompute_blocks=True)

        _log(
            f"fit start: problem_type={problem_type} X={X.shape} metric={eval_metric.name} "
            f"time_limit={time_limit} device={self._fit_resources['device']} num_cpus={num_cpus} "
            f"inner_splits={'custom ' + str(len(splits)) if splits else self.inner_folds}",
        )
        self._num_cpus = num_cpus
        with _torch_threads(num_cpus):
            self._pipeline = BloksPipeline(
                problem_type=problem_type,
                n_classes=n_classes,
                error_fn=make_error_fn(eval_metric, problem_type),
                bloks_cfg=bloks_cfg,
                hpo=self.hpo,
                n_trials=self.n_trials,
                n_startup_trials=self.n_startup_trials,
                inner_folds=self.inner_folds,
                ensemble_size=self.ensemble_size,
                pool_size=self.pool_size,
                max_onehot=self.max_onehot,
                time_limit=time_limit,
                time_safety=self.time_safety,
                default_budget_fraction=self.default_budget_fraction,
                random_state=random_state if random_state is not None else 0,
                log=_log,
            ).fit(X, y_enc, splits=splits)
        p = self._pipeline
        _log(
            f"fit done in {p.fit_time_s_:.1f}s: {len(p.history_)} trials, val_error={p.val_error_:.5f}, "
            f"ensemble of {len(p.members_)} configs, M={p.summary()['num_centers']}",
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self._pipeline is None:
            raise ValueError("System not yet fitted.")
        with _torch_threads(self._num_cpus):
            return pd.Series(self._pipeline.predict(X), index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._pipeline is None:
            raise ValueError("System not yet fitted.")
        # Columns are the original labels, like the AutoGluon system's predict_proba.
        with _torch_threads(self._num_cpus):
            proba = self._pipeline.predict(X)
        return pd.DataFrame(proba, index=X.index, columns=self._classes)

    def get_metric_error_val(self) -> float:
        """OOF error of the final (ensembled) bagged system on the inner CV."""
        if self._pipeline is None:
            raise ValueError("System not yet fitted.")
        return float(self._pipeline.val_error_)

    def get_validation_record(self) -> dict:
        """The inner splits this fit resolved (counts, clamps, structure), for the result record."""
        if self._validation_resolution is None:
            return {}
        record = self._validation_resolution.to_record()
        if self._pipeline is not None:
            record["num_bag_folds_fitted"] = self._pipeline.summary()["n_inner_splits"]
        return record

    def get_metadata(self) -> dict:
        return {**self._pipeline.summary(), **self._fit_resources} if self._pipeline is not None else {}

    def cleanup(self):
        self._pipeline = None
        try:
            

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass