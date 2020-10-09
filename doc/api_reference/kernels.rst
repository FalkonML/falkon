
falkon.kernels
==============

.. automodule:: falkon.kernels
.. py:currentmodule:: falkon.kernels


Kernel
------

.. autoclass:: falkon.kernels.kernel.Kernel
    :members:
    :private-members: _prepare, _apply, _finalize, _prepare_sparse, _apply_sparse, _decide_mm_impl, _decide_mmv_impl, _decide_dmmv_impl
    :special-members: __call__

.. autoclass:: falkon.kernels.keops_helpers.KeopsKernelMixin
    :members:


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