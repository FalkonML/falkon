.. falkon documentation master file, created by
   sphinx-quickstart on Thu Jun 18 10:38:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
Falkon
==================================
A Python library for large-scale kernel methods, with optional (multi-)GPU acceleration.

The library currently includes two solvers:
one for approximate kernel ridge regression :ref:`[2] <flk_2>` which is extremely fast, and one for kernel logistic
regression :ref:`[3] <log_flk>` which trades off lower speed for better accuracy on binary classification problems.

The main features of Falkon are:

 * *Full multi-GPU support* - All compute-intensive parts of the algorithms are multi-GPU capable.
 * *Extreme scalability* - Unlike other kernel solvers, we keep memory usage in check. We have tested the library with
   datasets of billions of points.
 * *Sparse data support*
 * *Scikit-learn integration* - Our estimators follow the scikit-learn API

For more details about the algorithms used, you can read :ref:`our paper <flk_1>`, or look at the source code
`github.com/FalkonML/falkon <https://github.com/falkonml/falkon>`_ and at the :ref:`documentation <api_reference>`.
Also, make sure to follow :ref:`the example notebooks <examples>` to find out about all of Falkon's features.

Falkon is built on top of `PyTorch <https://pytorch.org/>`__ which is used to support both CPU and GPU tensor
calculations, and `KeOps <https://www.kernel-operations.io/keops/index.html>`__ for fast kernel evaluations on the GPU.

If you find this library useful for your research, please cite our paper :ref:`[1] <flk_1>`!

Contents
========

.. toctree::
   :maxdepth: 2

   install
   get_started
   examples/examples
   api_reference/index


References
==========

.. _flk_1:

Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi,
"Kernel methods through the roof: handling billions of points efficiently,"
Advancs in Neural Information Processing Systems, 2020.

.. _flk_2:

Alessandro Rudi, Luigi Carratino, Lorenzo Rosasco, "FALKON: An optimal large scale kernel method,"
Advances in Neural Information Processing Systems, 2017.

.. _log_flk:

Ulysse Marteau-Ferey, Francis Bach, Alessandro Rudi, "Globally Convergent Newton Methods for Ill-conditioned
Generalized Self-concordant Losses," Advances in Neural Information Processing Systems, 2019.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
