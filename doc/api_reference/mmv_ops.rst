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


fmm_cpu
-------

Blocked kernel calculation on the CPU.

.. autofunction:: falkon.mmv_ops.fmm_cpu.fmm_cpu


fmm_cpu_sparse
--------------

Blocked kernel calculation on the CPU for sparse datasets.

.. autofunction:: falkon.mmv_ops.fmm_cpu.fmm_cpu_sparse


fmm_cuda
--------

Blocked kernel calculation on GPUs.

.. autofunction:: falkon.mmv_ops.fmm_cuda.fmm_cuda


fmm_cuda_sparse
---------------

Blocked kernel calculation on GPUs for sparse datasets.

.. autofunction:: falkon.mmv_ops.fmm_cuda.fmm_cuda_sparse


fmmv_cpu
---------------

Blocked kernel-vector products on CPU.

.. autofunction:: falkon.mmv_ops.fmmv_cpu.fmmv_cpu


fmmv_cpu_sparse
---------------

Blocked kernel-vector products on CPU for sparse datasets.

.. autofunction:: falkon.mmv_ops.fmmv_cpu.fmmv_cpu_sparse


fdmmv_cpu
---------------

Blocked double kernel-vector products on CPU.

.. autofunction:: falkon.mmv_ops.fmmv_cpu.fdmmv_cpu


fdmmv_cpu_sparse
----------------

Blocked double kernel-vector products on CPU for sparse datasets.

.. autofunction:: falkon.mmv_ops.fmmv_cpu.fdmmv_cpu_sparse


fmmv_cuda
----------------

Blocked kernel-vector products on GPUs.

.. autofunction:: falkon.mmv_ops.fmmv_cuda.fmmv_cuda


fmmv_cuda_sparse
----------------

Blocked kernel-vector products on GPUs for sparse datasets.

.. autofunction:: falkon.mmv_ops.fmmv_cuda.fmmv_cuda_sparse


fdmmv_cuda
----------------

Blocked double kernel-vector products on GPUs.

.. autofunction:: falkon.mmv_ops.fmmv_cuda.fdmmv_cuda


fdmmv_cuda_sparse
-----------------

Blocked double kernel-vector products on GPUs for sparse datasets.

.. autofunction:: falkon.mmv_ops.fmmv_cuda.fdmmv_cuda_sparse


incore_fmmv
-----------

.. autofunction:: falkon.mmv_ops.fmmv_incore.incore_fmmv


incore_fdmmv
------------

.. autofunction:: falkon.mmv_ops.fmmv_incore.incore_fdmmv

