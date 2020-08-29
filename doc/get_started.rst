.. _get_started:

Getting Started
===============

Once Falkon is installed, getting started is easy. The basic setup to use the `Falkon` estimator only requires
few lines of code:

.. code-block:: python

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


Passing Options
~~~~~~~~~~~~~~~

A number of different options exist for both the `Falkon` and `LogisticFalkon` estimators (see :ref:`falkon.Options <api_options>`).
All options can be passed to the estimator through the `FalkonOptions` class, like so:

.. code-block:: python

    from falkon import FalkonOptions, Falkon, kernels

    # Options to: increase the amount of output information; avoid using the KeOps library
    options = FalkonOptions(debug=True, no_keops=True)
    kernel = kernels.GaussianKernel(sigma=1.0)

    model = Falkon(kernel=kernel,
                   penalty=1e-6,
                   M=100,
                   maxiter=10,   # Set the maximum number of conjugate gradient iterations to 10
                   options=options)


More Examples
~~~~~~~~~~~~~

For more detailed examples, have a look at the :ref:`example notebooks <examples>`.