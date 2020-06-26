## Benchmark Scripts

This folder contains the code necessary to reproduce the benchmark results of the paper: [Kernel methods through the roof: handling billions of points efficiently](https://arxiv.org/abs/2006.10350).

It contains code for defining [GPyTorch](https://gpytorch.ai/) and [GPFlow](https://www.gpflow.org/) models, 
for data preprocessing (see the `datasets.py` file), and for running all standard benchmarks (see `benchmark_runner.py`).
The individual bash files are used as drivers which call the benchmark runner with different parameters.
The [EigenPro](https://github.com/EigenPro/EigenPro2) model code is missing from here, 
but is very similar to the publicly available code, and is available on request.


Other benchmarks are also run with scripts from this folder:
 - The out-of-core operation timings can be run with `potrf_timings.py` and `lauum_timings.py` and their respective drivers
 - The kernel matrix-vector multiplication experiment can be run with `mmv_timings.py`.
 - The experiment to measure timings with different features turned on is available in `time_improvements.py`.
