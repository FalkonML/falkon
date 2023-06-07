import glob
import os
import os.path as osp
import platform
import sys
from typing import Any, Tuple, List
from setuptools import setup, find_packages, Extension, dist

import numpy
import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = CUDA_HOME is not None or torch.version.hip
if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    WITH_CUDA = False
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"

WITH_CYTHON = False
try:
    from Cython.Build import cythonize
    WITH_CYTHON = True
except ImportError:
    cythonize = None


def get_version(root_dir):
    with open(os.path.join(root_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


def torch_version():
    import torch
    version = torch.__version__
    split_version = version.split(".")
    # With torch 1.10.0 the version 'number' include CUDA version (e.g. '1.10.0+cu102').
    # Here we remove the CUDA version.
    for i in range(len(split_version)):
        if '+' in split_version[i]:
            split_version[i] = split_version[i].split('+')[0]
    return [int(v) for v in split_version]


def torch_version_macros():
    int_version = torch_version()
    return [('TORCH_VERSION_MAJOR', int_version[0]),
            ('TORCH_VERSION_MINOR', int_version[1]),
            ('TORCH_VERSION_PATCH', int_version[2])]


def get_extensions():
    extensions = []

    # All C/CUDA routines are compiled into a single extension
    ext_cls = CppExtension
    ext_dir = osp.join('.', 'falkon', 'csrc')
    ext_files = (glob.glob(osp.join(ext_dir, '*.cpp')) +
                 glob.glob(osp.join(ext_dir, 'cpu', '*.cpp')))

    libraries = []
    macros: List[Tuple[str, Any]] = torch_version_macros()
    undef_macros = []
    extra_compile_args = {'cxx': ['-O3']}
    if not os.name == 'nt':  # Not on Windows:
        extra_compile_args['cxx'] += ['-Wno-sign-compare']
    if sys.platform == 'darwin':  # On macOS:
        extra_compile_args['cxx'] += ['-D_LIBCPP_DISABLE_AVAILABILITY']
    extra_link_args = [] if WITH_SYMBOLS else ['-s']

    info = parallel_info()
    if ('backend: OpenMP' in info and 'OpenMP not found' not in info
            and sys.platform != 'darwin'):
        extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
        if sys.platform == 'win32':
            extra_compile_args['cxx'] += ['/openmp']
        else:
            extra_compile_args['cxx'] += ['-fopenmp']
    else:
        print('Compiling without OpenMP...')

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    if WITH_CUDA:
        ext_cls = CUDAExtension
        cuda_files = (glob.glob(osp.join(ext_dir, 'cuda', '*.cu')) +
                      glob.glob(osp.join(ext_dir, 'cuda', '*.cpp')))
        ext_files.extend(cuda_files)
        macros.append(('WITH_CUDA', None))
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags.append('-O3')
        if torch.version.hip:
            # USE_ROCM was added to later versions of PyTorch
            # Define here to support older PyTorch versions as well:
            macros += [('USE_ROCM', None)]
            undef_macros += ['__HIP_NO_HALF_CONVERSIONS__']
        else:
            nvcc_flags += ['--expt-relaxed-constexpr', '--extended-lambda']
        extra_compile_args['nvcc'] = nvcc_flags
        extra_link_args += ['-lcusparse', '-l', 'cusparse',
                            '-lcublas', '-l', 'cublas',
                            '-lcusolver', '-l', 'cusolver',
                            '-ltorch_cuda_linalg', '-l', 'torch_cuda_linalg']
        libraries += ['cusolver', 'cublas', 'cusparse', 'torch_cuda_linalg']

    print(f"Defining C-extension on platform {sys.platform}. compile args: {extra_compile_args}  "
          f"macros: {macros}  link args: {extra_link_args}")
    # remove generated 'hip' files, in case of rebuilds
    ext_files = [path for path in ext_files if 'hip' not in path]

    extensions.append(
        ext_cls(
            "falkon.c_ext",
            sources=ext_files,
            include_dirs=[ext_dir],
            define_macros=macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        )
    )

    # Cyblas helpers
    file_ext = '.pyx' if WITH_CYTHON else '.c'
    extra_compile_args = ['-shared', '-fPIC', '-O3', '-Wall', '-std=c99']
    if 'OpenMP not found' not in info and sys.platform != 'darwin':
        extra_compile_args.append('-fopenmp')
    extra_link_args = ['-fPIC']
    print(f"Defining Cython extension on platform {sys.platform}. "
          f"compile args: {extra_compile_args}  link args: {extra_link_args}")
    cyblas_ext = [Extension('falkon.la_helpers.cyblas',
                            sources=[osp.join('falkon', 'la_helpers', 'cyblas' + file_ext)],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=extra_compile_args,
                            #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                            extra_link_args=extra_link_args)]
    if WITH_CYTHON:
        cyblas_ext = cythonize(cyblas_ext)
    extensions.extend(cyblas_ext)
    return extensions


# Requirements
install_requires = [
    'torch>=1.9',
    'scipy',
    'numpy',
    'scikit-learn',
    'psutil',
    'keopscore @ git+https://github.com/getkeops/keops.git@main#subdirectory=keopscore',
    'pykeops @ git+https://github.com/getkeops/keops.git@main#subdirectory=pykeops',
]
test_requires = [
    'pandas',
    'pytest',
    'pytest-cov',
    'coverage',
    'codecov',
    'flake8',
]
doc_requires = [
    'pandas',
    'numpydoc',
    'sphinx',
    'nbsphinx',
    'sphinx-rtd-theme',
    'matplotlib',
    'jupyter',
    'ghp-import',
    # Also pandoc, must be installed system-wide with apt
]

# Make sure we have numpy setup before attempting to run anything else.
# Numpy is actually needed only to get the include-directories for Cython.
# https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
dist.Distribution().fetch_build_eggs(['numpy'])

setup(
    name="falkon",
    version=get_version("falkon"),
    description="Fast, GPU enabled, approximate kernel ridge regression solver.",
    python_requires='>=3.7',
    tests_require=test_requires,
    extras_require={
        'test': test_requires,
        'doc': doc_requires
    },
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension.with_options(
            no_python_abi_suffix=True
        )
    },
    packages=find_packages(),
    include_package_data=True,  # Since we have a MANIFEST.in this will take all from there.
)
