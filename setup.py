import os
import os.path as osp

import numpy
from setuptools import setup, find_packages, Extension

try:
    import torch
except ImportError:
    raise ImportError("pytorch must be installed to setup Falkon.")
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, CppExtension
WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

try:
    from Cython.Build import cythonize
except:
    WITH_CYTHON = False
else:
    WITH_CYTHON = True


def get_version(root_dir):
    with open(os.path.join(root_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


def get_extensions():
    extensions = []

    # Sparse
    extension_cls = CppExtension
    sparse_ext_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'falkon', 'sparse')
    sparse_files = ['sparse_extension.cpp', 'cpp/sparse_matmul.cpp', 'cpp/sparse_norm.cpp']
    sparse_compile_args = {'cxx': ['-fopenmp']}
    sparse_link_args = []
    sparse_macros = []
    if WITH_CUDA:
        extension_cls = CUDAExtension
        sparse_files.extend(['cuda/csr2dense_cuda.cu', 'cuda/spspmm_cuda.cu'])
        sparse_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['--expt-relaxed-constexpr']
        sparse_compile_args['nvcc'] = nvcc_flags
        sparse_link_args += ['-lcusparse', '-l', 'cusparse']
    extensions.append(
        extension_cls("falkon.sparse.sparse_helpers",
                      sources=[osp.join(sparse_ext_dir, f) for f in sparse_files],
                      include_dirs=[sparse_ext_dir],
                      define_macros=sparse_macros,
                      extra_compile_args=sparse_compile_args,
                      extra_link_args=sparse_link_args,
                      )
    )

    # Parallel OOC
    if WITH_CUDA:
        ooc_ext_dir = osp.join(
            osp.dirname(osp.abspath(__file__)), 'falkon', 'ooc_ops', 'multigpu')
        ooc_files = ['multigpu_potrf_bind.cpp', 'cuda/multigpu_potrf.cu']
        ooc_macros = [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['--expt-relaxed-constexpr']
        ooc_compile_args = {'nvcc': nvcc_flags, 'cxx': []}
        ooc_link_args = ['-lcublas', '-l', 'cublas', '-lcusolver', '-l', 'cusolver']
        extensions.append(
            CUDAExtension(
                "falkon.ooc_ops.multigpu_potrf",
                sources=[osp.join(ooc_ext_dir, f) for f in ooc_files],
                include_dirs=[ooc_ext_dir],
                define_macros=ooc_macros,
                extra_compile_args=ooc_compile_args,
                extra_link_args=ooc_link_args,
                libraries=['cusolver', 'cublas'],
            )
        )

    # Cyblas helpers
    file_ext = '.pyx' if WITH_CYTHON else '.c'
    cyblas_compile_args = [
        '-shared', '-fPIC', '-fopenmp', '-O3', '-Wall']
    cyblas_ext = [Extension('falkon.utils.cyblas',
                            sources=['falkon/utils/cyblas' + file_ext],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=cyblas_compile_args,
                            extra_link_args=['-fPIC', '-fopenmp', '-s'])]
    if WITH_CYTHON:
        cyblas_ext = cythonize(cyblas_ext)
    extensions.extend(cyblas_ext)
    return extensions


install_requires = [
    'torch>=1.4',
    'scipy',
    'numpy',
    'scikit-learn',
    'psutil',
    'dataclasses;python_version<"3.7"',
]
test_requires = [
    'pytest',
    'sphinx',
    'nbsphinx',
    'sphinx-rtd-theme',
    'pandas',
    'matplotlib',
    'jupyter',
    # TODO: I'm sure there is more
]

setup(
    name="falkon",
    version=get_version("."),
    description="FALKON",
    python_requires='~=3.6',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'numpy',  # This is handled in pyproject.toml (since we're importing it at top of this file)
        'scipy',
    ],
    test_requires=test_requires,
    ext_modules=get_extensions(),
    packages=find_packages(),
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    install_requires=install_requires,
)
