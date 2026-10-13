from __future__ import annotations

from tabarena.utils.config_utils import SystemConfigGenerator
from falkon.benchmarks.tabarena.system2 import BloksSystemModel

# BLOKS does its own HPO and ensembling inside the time budget, so the generator only lists a few
# presets (no search space). They map onto TabArena's three model regimes:
#   BLOKS_c1: TPE HPO + greedy ensemble over the best bagged configs  ~ "tuned + ensemble"
#   BLOKS_c2: TPE HPO, best bagged config only                        ~ "tuned"
#   BLOKS_c3: the fixed default config, bagged                        ~ "default"
# The time budget comes from the run's resources (`build_experiments(time_limit=...)`), not from here.
gen_bloks = SystemConfigGenerator(
    model_cls=BloksSystemModel,
    name="BLOKS",
    manual_configs=[
        {},
        {"ensemble_size": 1},
        {"hpo": "none"},
    ],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_bloks.generate_all_system_experiments(
                num_random_configs=0,
            ), # type: ignore
        ),
    )