

falkon.kernels
==============

.. automodule:: falkon.kernels
.. py:currentmodule:: falkon.kernels


Kernel
------

.. autoclass:: falkon.kernels.kernel.Kernel
    :members:
    :private-members: _decide_mm_impl, _decide_mmv_impl, _decide_dmmv_impl
    :special-members: __call__

DiffKernel
----------

.. autoclass:: falkon.kernels.diff_kernel.DiffKernel
    :members: compute_diff, detach, diff_params

KeopsKernelMixin
----------------

.. autoclass:: falkon.kernels.keops_helpers.KeopsKernelMixin
    :members: keops_mmv, keops_mmv_impl


Radial kernels
--------------

Gaussian kernel
~~~~~~~~~~~~~~~

.. autoclass:: GaussianKernel
    :members: mmv, dmmv
    :special-members: __call__

Laplacian kernel
~~~~~~~~~~~~~~~~

.. autoclass:: LaplacianKernel
    :members: mmv, dmmv
    :special-members: __call__

Matern kernel
~~~~~~~~~~~~~

.. autoclass:: MaternKernel
    :members: mmv, dmmv
    :special-members: __call__


Dot-Product kernels
-------------------

Polynomial kernel
~~~~~~~~~~~~~~~~~

.. autoclass:: PolynomialKernel
    :members: mmv, dmmv
    :special-members: __call__

Linear kernel
~~~~~~~~~~~~~

.. autoclass:: LinearKernel
    :members: mmv, dmmv
    :special-members: __call__

Sigmoid kernel
~~~~~~~~~~~~~~~~~~

.. autoclass:: SigmoidKernel
    :members: mmv, dmmv
    :special-members: __call__
