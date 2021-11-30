falkon.mmv_ops
==============

The algorithms to compute kernels and kernel-vector products blockwise on GPUs and CPU. The algorithms in this module
are kernel agnostic. Refer to :mod:`falkon.kernels` for the actual kernel implementations.

The KeOps wrapper only supports the `mmv` operation (kernel-vector products). The matrix-multiplication implementations
instead support three different operations:

 - `mm` which calculates the full kernel
 - `mmv` which calculates kernel-vector products
 - `dmmv` which calculates double kernel-vector products (which are operations like :math:`K^\top (K v)` where
   :math:`K` is a kernel matrix and :math:`v` is a vector).

.. automodule:: falkon.mmv_ops
.. py:currentmodule:: falkon.mmv_ops

run_keops_mmv
-------------

A thin wrapper to KeOps is provided to allow for block-splitting and multiple GPU usage. This only supports
kernel-vector products.

.. autofunction:: falkon.mmv_ops.keops.run_keops_mmv


fmm
---

Block-wise kernel calculation. If the inputs require gradient, this function uses a differentiable implementation.

.. autofunction:: falkon.mmv_ops.fmm.fmm



fmmv
----

Block-wise kernel-vector products.

.. autofunction:: falkon.mmv_ops.fmmv.fmmv



fdmmv
-----

Block-wise double kernel-vector products.

.. autofunction:: falkon.mmv_ops.fmmv.fdmmv



incore_fmmv
-----------

.. autofunction:: falkon.mmv_ops.fmmv_incore.incore_fmmv


incore_fdmmv
------------

.. autofunction:: falkon.mmv_ops.fmmv_incore.incore_fdmmv


Low-level functions
-------------------

The following are some of the low-level functions which help compute kernels and kernel-vector products block-wise.
They are specialized for different input types.

.. autofunction:: falkon.mmv_ops.fmm.sparse_mm_run_thread

.. autofunction:: falkon.mmv_ops.fmmv.sparse_mmv_run_thread
