# Gradient-Based Hyperparameter Optimization Module

The most interesting code for users lies in the `objectives` submodule, 
which contains a collection of several optimization objectives 
(i.e. penalized losses), which can be used to optimize the hyperparameters
of (Nystrom) kernel ridge regression.

There are several exact objectives, which typically require storing the 
full K_{nm} kernel matrix in memory, and are therefore suited for medium size
problems.
One approximate objective is also implemented (the `StochasticNystromCompReg`
class), which can scale up to much larger problems. This stochastic objective
might be somewhat hard to tune in order to obtain good performmance.

The `optimization` submodule contains helpers for gradient-based optimization,
error / performance reports, and a grid-search implementation.

The `benchmarking` submodule contains a large runner with several parameters which
has been used to run experiments on hyperparameter optimization.
