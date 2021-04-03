falkon.ooc_ops
==============

The out-of-core algorithms for the Cholesky decomposition and the LAUUM operation are crucial for speeding up our
library. To find out more about how they work, check the source code:

 - `Out of core Cholesky <https://github.com/FalkonML/falkon/blob/master/falkon/ooc_ops/multigpu/cuda/multigpu_potrf.cu>`_ (CUDA code)
 - `Out of core LAUUM <https://github.com/FalkonML/falkon/blob/master/falkon/ooc_ops/parallel_lauumm.py>`_ (Python code)

The following functions provide a higher-level interface to the two operations.

.. automodule:: falkon.ooc_ops
.. py:currentmodule:: falkon.ooc_ops


gpu_cholesky
------------

.. autofunction:: gpu_cholesky


gpu_lauum
---------

.. autofunction:: gpu_lauum
