.. _install:

Install
=======

Supported Platforms
-------------------

Falkon is only tested on Linux.

GPU support is achieved via CUDA, so using Falkon from Windows will be tricky.

Using the library from the CPU only is much more portable since there is no CUDA requirement, and it will likely work
on Windows and Mac OS more easily.

Prerequisites
-------------

PyTorch and CUDA
~~~~~~~~~~~~~~~~
Falkon depends on PyTorch, and on the NVIDIA Toolkit (for NVIDIA GPU support) which is usually installed
alongside PyTorch.
PyTorch can be installed in several ways by following the instructions at `Install PyTorch <https://pytorch.org/get-started/locally/>`__.
If GPU support is desired, make sure that PyTorch CUDA bindings are working:

.. code-block:: python

    import torch
    assert torch.cuda.is_available()

Intel MKL
~~~~~~~~~
If Falkon is not installed with GPU support, it will try to link to `Intel MKL <https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html>`__
to speed-up certain sparse operations. MKL shared libraries are usually distributed with numpy, so this should not be a problem.
In case sparse matrix support is not needed, the MKL library will not be loaded.



Installing
----------

There are three ways of installing Falkon:

1. **From source** by running

    .. code-block:: bash

      $ pip install --no-build-isolation git+https://github.com/falkonml/falkon.git

2. **From pypi with JIT compilation** (the C++ extension will be compiled when the library is first used) **BROKEN!!**:

    .. code-block:: bash

      $ pip install falkon

3. **From pre-built wheels** which are available for the following versions of PyTorch and CUDA:

     ============== ========= ========= ========= =========
      Linux          `cu116`   `cu117`   `cu118`   `cu121`
     ============== ========= ========= ========= =========
      torch 1.13.0    ✅        ✅
      torch 2.0.0               ✅        ✅
      torch 2.1.0                         ✅        ✅
     ============== ========= ========= ========= =========

    As an example, if you **already have installed** PyTorch 1.13 and CUDA 11.7 on your system, you should run

    .. code-block:: bash

      $ pip install falkon -f https://falkon.dibris.unige.it/torch-1.13.0_cu117.html

    Similarly for **CPU-only packages**

    .. code-block:: bash

      $ pip install falkon -f https://falkon.dibris.unige.it/torch-2.0.0_cpu.html

    please check `here <https://falkon.dibris.unige.it/index.html>`__ for a list of supported wheels.

For options 1 and 2, you will need the CUDA toolkit to be setup properly on your system in order to compile the sources.
Compilation may take a few minutes. To speed it up you can try to install ``ninja`` (``pip install ninja``) which
parallelizes the build process.


Testing the installation
------------------------

To check that everything works correctly you can follow the `Kernel ridge regression <examples/falkon_regression_tutorial.ipynb>`_ notebook.



Development
-----------

For development purposes the library should be installed in editable mode (i.e. `pip install -e .` from the
falkon directory.

To build the documentation go into the `doc` directory and run `make html`.
