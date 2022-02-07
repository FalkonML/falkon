import glob
import os
import os.path as osp
import platform
import sys
from typing import Any, Tuple, List

import numpy
from setuptools import setup, find_packages, Extension

try:
    import torch
except ImportError:
    raise ImportError("PyTorch must be pre-installed before installing Falkon.")
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, CppExtension

ONLY_CPU = os.getenv('FORCE_ONLY_CPU', '0')
FORCE_CUDA = os.getenv('FORCE_CUDA', '0')
WITH_CUDA = (
    (torch.cuda.is_available() and CUDA_HOME is not None and ONLY_CPU != '1') or
    (FORCE_CUDA == '1')
)

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
    WITH_CYTHON = False
else:
    WITH_CYTHON = True


CURRENT_DIR = "."


def get_version(root_dir):
    with open(os.path.join(root_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


def parallel_backend():
    # https://github.com/suphoff/pytorch_parallel_extension_cpp/blob/master/setup.py
    from torch.__config__ import parallel_info
    parallel_info_string = parallel_info()
    parallel_info_array = parallel_info_string.splitlines()
    backend_lines = [line for line in parallel_info_array if line.startswith('ATen parallel backend:')]
    if len(backend_lines) != 1:
        return None
    backend = backend_lines[0].rsplit(': ')[1]
    return backend


def parallel_extra_compile_args(is_torch: bool):
    if is_torch:
        backend = parallel_backend()
        if backend == 'OpenMP':
            p_args = ['-DAT_PARALLEL_OPENMP']
            if sys.platform == 'win32':
                p_args.append('/openmp')
            else:
                p_args.append('-fopenmp')
            return p_args
        elif backend == 'native thread pool':
            return ['-DAT_PARALLEL_NATIVE']
        elif backend == 'native thread pool and TBB':
            return ['-DAT_PARALLEL_NATIVE_TBB']
        else:
            return []
    else:
        info = torch.__config__.parallel_info()
        if 'OpenMP not found' not in info and sys.platform != 'darwin':
            if sys.platform == 'win32':
                return ['/openmp']
            else:
                return ['-fopenmp']
        else:
            return []


def torch_version():
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
    extension_cls = CppExtension
    ext_dir = osp.join(CURRENT_DIR, 'falkon', 'csrc')
    ext_files = (glob.glob(osp.join(ext_dir, '*.cpp')) +
                 glob.glob(osp.join(ext_dir, 'cpu', '*.cpp')))

    extra_compile_args = {
        'cxx': parallel_extra_compile_args(is_torch=True)
    }
    link_args = []
    libraries = []
    macros: List[Tuple[str, Any]] = torch_version_macros()

    # Compile for mac arm64
    if (sys.platform == 'darwin' and platform.machine() == 'arm64'):
        extra_compile_args['cxx'] += ['-arch', 'arm64']
        link_args += ['-arch', 'arm64']

    if WITH_CUDA:
        extension_cls = CUDAExtension
        cuda_files = (glob.glob(osp.join(ext_dir, 'cuda', '*.cu')) +
                      glob.glob(osp.join(ext_dir, 'cuda', '*.cpp')))
        ext_files.extend(cuda_files)
        macros.append(('WITH_CUDA', None))
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['--expt-relaxed-constexpr', '--expt-extended-lambda', '-O2']
        extra_compile_args['nvcc'] = nvcc_flags
        link_args += ['-lcusparse', '-l', 'cusparse',
                      '-lcublas', '-l', 'cublas',
                      '-lcusolver', '-l', 'cusolver']
        libraries.extend(['cusolver', 'cublas', 'cusparse'])

    extensions.append(
        extension_cls(
            "falkon.c_ext",
            sources=ext_files,
            include_dirs=[ext_dir],
            define_macros=macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=link_args,
            libraries=libraries,
        )
    )

    # Cyblas helpers
    file_ext = '.pyx' if WITH_CYTHON else '.c'
    extra_compile_args = parallel_extra_compile_args(is_torch=False)
    extra_compile_args += ['-shared', '-fPIC', '-O3', '-Wall', '-std=c99']
    extra_link_args = ['-fPIC']
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
    'dataclasses;python_version<"3.7"',
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

extras = {
    'test': test_requires,
    'doc': doc_requires
}

setup(
    name="falkon",
    version=get_version("falkon"),
    description="Fast, GPU enabled, approximate kernel ridge regression solver.",
    python_requires='~=3.7',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'numpy',
    ],
    tests_require=test_requires,
    extras_require=extras,
    ext_modules=get_extensions(),
    packages=find_packages(),
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    },
    install_requires=install_requires,
    include_package_data=True,  # Since we have a MANIFEST.in this will take all from there.
)
