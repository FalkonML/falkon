import glob
import os
import os.path as osp

import numpy
from setuptools import setup, find_packages, Extension

try:
    import torch
except ImportError:
    raise ImportError("pytorch must be pre-installed to setup Falkon.")
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME, CppExtension
WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

try:
    from Cython.Build import cythonize
except ImportError:
    WITH_CYTHON = False
else:
    WITH_CYTHON = True


CURRENT_DIR = "."  # osp.dirname(__file__)


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


def parallel_extra_compile_args():
    backend = parallel_backend()
    if (backend == 'OpenMP'):
        return ['-DAT_PARALLEL_OPENMP', '-fopenmp']
    elif (backend == 'native thread pool'):
        return ['-DAT_PARALLEL_NATIVE']
    elif (backend == 'native thread pool and TBB'):
        return ['-DAT_PARALLEL_NATIVE_TBB']
    return []


def get_extensions():
    extensions = []

    # All C/CUDA routines are compiled into a single extension
    extension_cls = CppExtension
    ext_dir = osp.join(CURRENT_DIR, 'falkon', 'csrc')
    ext_files = [osp.join(ext_dir, 'pytorch_bindings.cpp')]
    ext_files.extend(glob.glob(osp.join(ext_dir, 'cpp', '*.cpp')))
    compile_args = {'cxx': parallel_extra_compile_args()}
    link_args = []
    macros = []
    libraries = []
    if WITH_CUDA:
        extension_cls = CUDAExtension
        ext_files.extend(glob.glob(osp.join(ext_dir, 'cuda', '*.cu')))
        macros.append(('WITH_CUDA', None))
        nvcc_flags = os.getenv('NVCC_FLAGS', '')
        nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
        nvcc_flags += ['--expt-relaxed-constexpr']
        compile_args['nvcc'] = nvcc_flags
        link_args += ['-lcusparse', '-l', 'cusparse',
                      '-lcublas', '-l', 'cublas',
                      '-lcusolver', '-l', 'cusolver']
        libraries.extend(['cusolver', 'cublas', 'cusparse'])
    extensions.append(
        extension_cls("falkon.c_ext",
                      sources=ext_files,
                      include_dirs=[ext_dir],
                      define_macros=macros,
                      extra_compile_args=compile_args,
                      extra_link_args=link_args,
                      libraries=libraries,
                      )
    )

    # Cyblas helpers
    file_ext = '.pyx' if WITH_CYTHON else '.c'
    cyblas_compile_args = [
        '-shared', '-fPIC', '-fopenmp', '-O3', '-Wall', '-std=c99']
    cyblas_ext = [Extension('falkon.la_helpers.cyblas',
                            sources=[osp.join('falkon', 'la_helpers', 'cyblas' + file_ext)],
                            include_dirs=[numpy.get_include()],
                            extra_compile_args=cyblas_compile_args,
                            #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                            extra_link_args=['-fPIC', '-fopenmp', '-s'])]
    if WITH_CYTHON:
        cyblas_ext = cythonize(cyblas_ext)
    extensions.extend(cyblas_ext)
    return extensions

# Requirements -- TODO: We also have requirements.txt files lying around which are out of sync.
install_requires = [
    'torch>=1.4',
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
    # There is also pandoc O.o
]

extras = {
    'test': test_requires,
    'doc': doc_requires
}

setup(
    name="falkon",
    version=get_version("falkon"),
    description="FALKON",
    python_requires='~=3.6',
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
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    install_requires=install_requires,
    include_package_data=True,  # Since we have a MANIFEST.in this will take all from there.
)
