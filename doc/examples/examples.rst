.. _examples:

Examples
========

.. toctree::
    :maxdepth: 1
    :hidden:

    ./falkon_regression_tutorial.ipynb
    ./logistic_falkon.ipynb
    ./falkon_cv.ipynb
    ./custom_kernels.ipynb
    ./hyperopt.ipynb
    ./falkon_mnist.ipynb

.. _Kernel ridge regression:
    ./falkon_regression_tutorial.ipynb

.. _Logistic Falkon tutorial:
    ./logistic_falkon.ipynb

.. _Hyperparameter tuning:
    ./falkon_cv.ipynb

.. _custom kernels:
    ./custom_kernels.ipynb

.. _Gradient hyperopt:
    ./hyperopt.ipynb

.. _MNIST example:
    ./falkon_mnist.ipynb


Starting with simple kernel ridge regression, via classification, hyperparameter tuning, to large-scale GPU experiments,
these notebooks cover all there is to know about Falkon.

 - `Kernel ridge regression`_ goes through the basic notions of the library with a simple example;
 - `Logistic Falkon tutorial`_ shows how to use the Logistic Falkon estimator, comparing the results with normal Falkon;
 - `Hyperparameter tuning`_ is a fully worked out example of optimizing hyperparameters with cross-validation for a real-world multi-class problem;
 - `custom kernels`_ will walk you through the implementation of a custom kernel.
 - `Gradient hyperopt`_: a tutorial on using the :mod:`~falkon.hopt` module for gradient-based hyperparameter optimization in Falkon.
 - `MNIST example`_: A simple tutorial on using Falkon for MNIST digit classification.

