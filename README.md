# FALKON

Python implementation of the Falkon algorithm for large-scale, approximate kernel ridge regression.

This repository is well optimized and can scale to problems with tens of millions of points. Full kernel matrices are never computed explicitly so that you will not run out of memory on large problems.

Preconditioned conjugate gradient optimization ensures that only few iterations are necessary to obtain good results.

The basic algorithm is a Nyström approximation to KRR, which needs only three hyperparameters:
 1. The number of centers `M` - this controls the quality of the approximation: a higher number of centers will produce more accurate results at the expense of more computation time.
 Setting `M = sqrt(N)` where `N` is the number of data-points has been shown to guarantee the same theoretical accuracy as exact KRR. On certain points it may be advantageous to further increase `M`.
 2. The penalty term λ (argument `la` below) controls the amount of regularization. Higher λ tends to prevent overfitting but, depending on the problem, a lower λ may give more accurate results.
 3. The kernel function. A good default is always the Gaussian kernel (`falkon.kernels.GaussianKernel`).


## Documentation

Extensive documentation is available at [https://falkonml.github.io/falkon/]. Several worked-through examples
are also provided in [the docs](https://falkonml.github.io/falkon/examples/examples.html).

If you find a bug, please open a new issue on GitHub!


## Reference

If you find this library useful for your work, please cite the following publication:
```
@misc{falkonlibrary2020,
    title={Kernel methods through the roof: handling billions of points efficiently},
    authors={Meanti, Giacomo and Carratino, Luigi and Rosasco, Lorenzo and Rudi, Alessandro},
	year = {2020},
	archivePrefix = {arXiv},
	eprint = {2006.10350}
}
```
