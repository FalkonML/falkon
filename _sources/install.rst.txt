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

KeOps
~~~~~
Falkon uses a patched version of `KeOps <https://www.kernel-operations.io/keops/index.html>`__, therefore until the
patches are merged in the KeOps library, users will need to uninstall existing KeOps installations (if present), and
install our patched version.

The patched version is available as a submodule of the Falkon repository. To install it, you can do the following:

.. code-block:: bash

    $ git clone --recurse-submodules https://github.com/FalkonML/falkon.git
    $ cd falkon/
    $ pip install ./keops

For use with **Google Colab** KeOps needs some additional packages. You can install them alongside KeOps by running

.. code-block::

    !pip install pykeops[colab] > install.log

from within the Jupyter notebook (see the `KeOps docs <https://www.kernel-operations.io/keops/python/installation.html>`__
for more information).

Installing
----------

After having installed KeOps, you are ready to install Falkon with `pip` (from the root of the git tree):

.. code-block:: bash

   $ pip install .

This will take a while since there are both Cython extensions which need to be compiled (if Cython is installed),
and PyTorch extensions which take some time to compile.


Testing the installation
------------------------

To check that everything works correctly you can follow the `Kernel ridge regression <examples/falkon_regression_tutorial.ipynb>`_ notebook.



Development
-----------

For development purposes the library should be installed in editable mode (i.e. `pip install -e .` from the
falkon directory.

To build the documentation go into the `doc` directory and run `make html`.
