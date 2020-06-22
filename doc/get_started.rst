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


For more detailed examples, have a look at the :ref:`example notebooks <examples>`.