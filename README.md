# FALKON

Python implementation of the Falkon algorithm for large-scale, approximate kernel ridge regression.

This repository is well optimized and can scale to problems with tens of millions of points. Full kernel matrices are never computed explicitly so that you will not run out of memory on large problems.

Preconditioned conjugate gradient optimization ensures that only few iterations are necessary to obtain good results.

The basic algorithm is a Nyström approximation to KRR, which needs only three hyperparameters:
 1. The number of centers `M` - this controls the quality of the approximation: a higher number of centers will produce more accurate results at the expense of more computation time.
 Setting `M = sqrt(N)` where `N` is the number of data-points has been shown to guarantee the same theoretical accuracy as exact KRR. On certain points it may be advantageous to further increase `M`.
 2. The penalty term λ (argument `la` below) controls the amount of regularization. Higher λ tends to prevent overfitting but, depending on the problem, a lower λ may give more accurate results.
 3. The kernel function. A good default is always the Gaussian kernel (`falkon.kernels.GaussianKernel`).

## Installation
The library has a few dependencies, the largest of which is [PyTorch](https://pytorch.org/) which needs to be installed **before** installing falkon.
Falkon has only been tested on Linux. It will likely work on Mac OS, unknown whether it would work on Windows.

1. Please make sure that [pykeops](http://www.kernel-operations.io/) is **not** already installed on your system as we will install 
a slightly modified version of pykeops below. If you had previously installed pykeops, uninstall it before proceeding further.
2. Install pytorch 1.4
3. Clone the FALKON repository
    ```bash
    git clone https://github.com/alessandro-rudi/FALKON.git
    cd FALKON
    git submodule update --init --recursive
    ```
4. Install pykeops
    ```bash
    pip install ./keops/
    ```
5. Install falkon 
    ```bash
    pip install .
    ```

## Core API
The core of the code is a single estimator which follows the scikit-learn API conventions:

```python
import torch
from sklearn.datasets import load_boston
from falkon import Falkon, kernels

X, Y = load_boston(return_X_y=True)
X = torch.from_numpy(X)
Y = torch.from_numpy(Y).reshape(-1, 1)

kernel = kernels.GaussianKernel(sigma=1.0)
model = Falkon(
    kernel=kernel,
    penalty=1e-6,
    M=100,
)
model.fit(X, Y)
predictions = model.predict(X)
```

where data (`x_tr`, `y_tr`, etc.) should be `torch.Tensor` objects.

Many additional options are available to control the behaviour of the algorithm more finely. See for example the later section on using the GPU.

A Jupyter notebook is available which shows in detail an example of [how to do regression](notebooks/FalkonRegression.ipynb) with Falkon.


## Error monitoring

When dealing with large problems you may want to monitor how the validation error evolves after each iteration.
This is possible with the Falkon library, although it requires deviating from the sklearn API.

It requires passing two additional arguments at model initialization:
 - `error_fn` should be a function which takes two arguments (`y_true` and `y_pred`) and outputs a `float` giving the error of the input predictions.
 For finer control this function may also return a tuple<str, float> where the first element is the name of the error (such as "MSE", "AUC", etc.)
 - `error_every` should be an integer which allows to control how often the validation error will be printed. With `error_every=0` you will print validation information at every iteration.

Additionally the fit method should take the validation data which is used to display metrics (arguments `Xts` and `Yts`).

## GPU Usage

The code can take advantage of existing GPUs. It will by default use all available GPUs to distribute the workloads. 
Restrict it by setting the `CUDA_VISIBLE_DEVICES` environment variable if necessary.
If a GPU is available but you don't want to use it, you can also pass `use_cpu=True` to the model. when using the CPU you may want to limit the amount of RAM used, with the `max_cpu_mem` option (specified in bytes).
When using the GPU, a similar option `max_gpu_mem` is available although given the usually low amount of GPU memory present, changing this option is not advised.


## Reference

If you find this library useful for your work, please cite the following publication:
```
@inproceedings{rudi2017falkon,
    title="{FALKON}: An Optimal Large Scale Kernel Method",
    authors="Rudi, Alessandro and Carratino, Luigi and Rosasco, Lorenzo",
    booktitle={Advances in Neural Information Processing Systems 29},
    pages={3888--3898},
    year={2017}
}
```
