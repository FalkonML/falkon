import glob
import os
import os.path as osp
import platform
import sys
from typing import Any, Tuple, List
from setuptools import setup, find_packages

import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (
    CUDA_HOME,
    TORCH_LIB_PATH,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

if os.getenv("FORCE_ONLY_CPU", "0") == "1":
    WITH_CUDA = False
elif CUDA_HOME is not None:
    WITH_CUDA = True
else:
    WITH_CUDA = False
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
NO_BUILD_EXT = os.getenv("NO_BUILD_EXT", "0") == "1"  # Don't build the extension at all (for JIT compilation)


def get_version(root_dir):
    with open(os.path.join(root_dir, "VERSION")) as version_file:
        version = version_file.read().strip()
    return version


def torch_version():
    import torch

    version = torch.__version__
    split_version = version.split(".")
    # With torch 1.10.0 the version 'number' include CUDA version (e.g. '1.10.0+cu102').
    # Here we remove the CUDA version.
    for i in range(len(split_version)):
        if "+" in split_version[i]:
            split_version[i] = split_version[i].split("+")[0]
    return [int(v) for v in split_version]


def torch_version_macros():
    int_version = torch_version()
    return [
        ("TORCH_VERSION_MAJOR", int_version[0]),
        ("TORCH_VERSION_MINOR", int_version[1]),
        ("TORCH_VERSION_PATCH", int_version[2]),
    ]


def get_extensions():
    extensions = []

    # All C/CUDA routines are compiled into a single extension
    ext_cls = CppExtension
    ext_dir = osp.join(".", "falkon", "c_ext")
    ext_files = (
        glob.glob(osp.join(ext_dir, "ops", "cpu", "*.cpp"))
        + glob.glob(osp.join(ext_dir, "ops", "autograd", "*.cpp"))
        + glob.glob(osp.join(ext_dir, "ops", "*.cpp"))
        + glob.glob(osp.join(ext_dir, "*.cpp"))
    )

    libraries = []
    macros: List[Tuple[str, Any]] = torch_version_macros()
    undef_macros = []
    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    if sys.platform == "darwin":  # On macOS:
        extra_compile_args["cxx"] += ["-D_LIBCPP_DISABLE_AVAILABILITY"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if "backend: OpenMP" in info and "OpenMP not found" not in info and sys.platform != "darwin":
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    if WITH_CUDA:
        ext_cls = CUDAExtension
        ext_files.extend(glob.glob(osp.join(ext_dir, "ops", "cuda", "*.cu")))
        macros.append(("WITH_CUDA", None))
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags.append("-O3")
        if torch.version.hip:
            # USE_ROCM was added to later versions of PyTorch
            # Define here to support older PyTorch versions as well:
            macros += [("USE_ROCM", None)]
            undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        else:
            nvcc_flags += ["--expt-relaxed-constexpr", "--extended-lambda"]
        extra_compile_args["nvcc"] = nvcc_flags
        extra_link_args += [
            "-L",
            os.path.join(CUDA_HOME, "lib"),
            "-L",
            TORCH_LIB_PATH,
            "-Wl,-rpath,$ORIGIN/../../torch/lib",
        ]
        libraries += ["cusolver", "cublas", "cusparse"]
        if torch.__version__ >= (1, 12):
            libraries.append("torch_cuda_linalg")

    print(
        f"Defining C-extension on platform {sys.platform}. compile args: {extra_compile_args}  "
        f"macros: {macros}  link args: {extra_link_args}  libraries {libraries}"
    )
    # remove generated 'hip' files, in case of rebuilds
    ext_files = [path for path in ext_files if "hip" not in path]

    extensions.append(
        ext_cls(
            "falkon.c_ext._C",
            sources=ext_files,
            include_dirs=[ext_dir],
            define_macros=macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        )
    )
    return extensions


# Requirements
install_requires = [
    "torch>=2.4",
    "scipy",
    "numpy",
    "scikit-learn",
    "wheel",
    "psutil",
    "keopscore>=2.2",
    "pykeops>=2.2",
]
test_requires = [
    "pandas",
    "pytest",
    "pytest-cov",
    "coverage[toml]",
    "codecov",
    "flake8",
]
doc_requires = [
    "pandas",
    "numpydoc",
    "sphinx",
    "nbsphinx",
    "sphinx-rtd-theme",
    "matplotlib",
    "jupyter",
    "ghp-import",
    # Also pandoc, must be installed system-wide with apt
]

setup(
    name="falkon",
    version=get_version("falkon"),
    author="Giacomo Meanti",
    author_email="giacomo.meanti@iit.it",
    url="https://falkonml.github.io/falkon/",
    description="Fast, GPU enabled, approximate kernel ridge regression solver.",
    python_requires=">=3.8",
    tests_require=test_requires,
    extras_require={"test": test_requires, "doc": doc_requires},
    install_requires=install_requires,
    ext_modules=get_extensions() if not NO_BUILD_EXT else [],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    } if not NO_BUILD_EXT else {},
    packages=find_packages(where="."),
    # Files in MANIFEST.in are included in sdist and in wheel only if include_package_data is True
    include_package_data=True,
    exclude_package_data={
        "falkon.c_ext": [
            "*.cpp",
            "*.h",
            "*.cu",
            "ops/*.cpp",
            "ops/*.h",
            "ops/*.cu",
            "ops/autograd/*.cpp",
            "ops/autograd/*.cu",
            "ops/autograd/*.h",
            "ops/cpu/*.cpp",
            "ops/cpu/*.cu",
            "ops/cpu/*.h",
            "ops/cuda/*.cpp",
            "ops/cuda/*.cu",
            "ops/cuda/*.h",
            "ops/cuda/*.cuh",
        ]
    },
)
