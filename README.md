[![](https://codecov.io/gh/FalkonML/falkon/branch/master/graphs/badge.svg?branch=master)](https://codecov.io/gh/FalkonML/falkon/)

# Falkon

Python implementation of the Falkon algorithm for large-scale, approximate kernel ridge regression.

The code is optimized for scalability to large datasets with tens of millions of points and beyond.
Full kernel matrices are never computed explicitly so that you will not run out of memory on larger problems.

Preconditioned conjugate gradient optimization ensures that only few iterations are necessary to obtain good results.

The basic algorithm is a Nyström approximation to kernel ridge regression, which needs only three hyperparameters:
 1. The number of centers `M` - this controls the quality of the approximation: a higher number of centers will
    produce more accurate results at the expense of more computation time, and higher memory requirements.
 2. The penalty term, which controls the amount of regularization.
 3. The kernel function. A good default is always the Gaussian (RBF) kernel
    ([`falkon.kernels.GaussianKernel`](https://falkonml.github.io/falkon/api_reference/kernels.html#gaussian-kernel)).

For more information about the algorithm and the optimized solver, you can download our paper:
[Kernel methods through the roof: handling billions of points efficiently](https://arxiv.org/abs/2006.10350)

## Documentation

The API is sklearn-like, so that Falkon should be easy to integrate in your existing code. 
Several worked-through examples are provided in [the docs](https://falkonml.github.io/falkon/examples/examples.html),
and for more advanced usage, extensive documentation is available at https://falkonml.github.io/falkon/.

If you find a bug, please open a new issue on GitHub!


## Installation

**Dependencies**: 
 - Please install [PyTorch](https://pytorch.org/get-started/locally/) first.
 - `cmake` and a C++ compiler are also needed for [KeOps](https://www.kernel-operations.io/keops/python/installation.html) acceleration (optional but strongly recommended).

To install from source, you can run

```bash
pip install --no-build-isolation git+https://github.com/FalkonML/falkon.git
```

We alternatively provide pre-built pip wheels for the following combinations of PyTorch and CUDA:

| Linux        | `cu117` | `cu118` | `cu121` |
|--------------|---------|---------|---------|
| torch 2.0.0  | ✅      | ✅      |         |
| torch 2.1.0  |         | ✅      | ✅      |
| torch 2.2.0  |         | ✅      | ✅      |

For other combinations, and previous versions of Falkon, please check [here](https://falkon.dibris.unige.it/index.html)
for a list of supported wheels.

To install a wheel for a specific PyTorch + CUDA combination, you can run
```bash
# e.g., torch 2.2.0 + CUDA 12.1
pip install falkon -f https://falkon.dibris.unige.it/torch-2.2.0_cu121.html
```

Similarly for CPU-only packages
```bash
# e.g., torch 2.1.0 + cpu
pip install falkon -f https://falkon.dibris.unige.it/torch-2.1.0_cpu.html
```

**More detailed installation instructions are available in [the documentation](https://falkonml.github.io/falkon/install.html).**

## Gradient-based Hyperparameter Optimization

**New!** We added a new module for automatic hyperparameter optimization. 
The [`falkon.hopt`](https://falkonml.github.io/falkon/api_reference/hopt.html) module contains the implementation
of several objective functions which can be minimized with respect to Falkon's hyperparameters (notably the penalty, 
the kernel parameters and the centers themselves).

For more details check out our paper: 
[Efficient Hyperparameter Tuning for Large Scale Kernel Ridge Regression](http://arxiv.org/abs/2201.06314),
and the [automatic hyperparameter tuning notebook](https://falkonml.github.io/falkon/examples/hyperopt.html).

## References

If you find this library useful for your work, please cite the following publications:
```
@inproceedings{falkonlibrary2020,
    title = {Kernel methods through the roof: handling billions of points efficiently},
    author = {Meanti, Giacomo and Carratino, Luigi and Rosasco, Lorenzo and Rudi, Alessandro},
    year = {2020},
    booktitle = {Advances in Neural Information Processing Systems 32}
}
```
```
@inproceedings{falkonhopt2022,
    title = {Efficient Hyperparameter Tuning for Large Scale Kernel Ridge Regression},
    author = {Meanti, Giacomo and Carratino, Luigi and De Vito, Ernesto and Rosasco, Lorenzo},
    year = {2022},
    booktitle = {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics}
}
```
