.. _examples:

Examples
========

Starting with simple kernel ridge regression, via classification, hyperparameter tuning, to large-scale GPU experiments,
these notebooks cover all there is to know about Falkon.

 - `Kernel ridge regression`_ goes through the basic notions of the library with a simple example;
 - `Logistic Falkon tutorial`_ shows how to use the Logistic Falkon estimator, comparing the results with normal Falkon;
 - `Hyperparameter tuning`_ is a fully worked out example of parameter optimization with Falkon for a real-world multi-class problem;
 - `GPU training`_ details the options to maximize performance on multiple GPU systems.

.. toctree::
    :maxdepth: 1
    :hidden:

    ./simple_regression.ipynb
    ./logistic_falkon.ipynb
    ./hyperparameter_tuning.ipynb
    ./performance_tuning.ipynb


.. _Kernel ridge regression:
    ./simple_regression.ipynb

.. _Logistic Falkon tutorial:
    ./logistic_falkon.ipynb

.. _Hyperparameter tuning:
    ./hyperparameter_tuning.ipynb

.. _GPU training:
    ./performance_tuning.ipynb
