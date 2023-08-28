"""
Taken from nerfacc (https://github.com/KAIR-BAIR/nerfacc) (MIT Licence)

Copyright (c) 2022 Ruilong Li, UC Berkeley.
Copyright (c) 2023 Giacomo Meanti
"""

import glob
import importlib.machinery
import json
import os
import os.path as osp
import shutil
import warnings
from subprocess import DEVNULL, call
from typing import Optional

import torch.cuda
from torch.utils.cpp_extension import _get_build_directory, load


def _get_extension_path(lib_name):
    lib_dir = os.path.dirname(__file__)
    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError

    return ext_specs.origin


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    """Get the cuda toolkit version."""
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


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


def lib_from_oserror(exc: OSError) -> Optional[str]:
    e_str = str(exc)
    so_idx = e_str.index(".so")  # TODO: Platform specific code
    if so_idx <= 0:
        return None
    lib_wext = e_str[: so_idx + 3]
    return lib_wext


_HAS_EXT = False

try:
    if not _HAS_EXT:
        # try to import the compiled module (via setup.py)
        lib_path = _get_extension_path("_C")
        try:
            torch.ops.load_library(lib_path)
        except OSError as e:
            # Hack: usually ld can't find torch_cuda_linalg.so which is in TORCH_LIB_PATH
            # if we load it first, then load_library will work.
            # TODO: This will only work on linux.
            if (missing_lib := lib_from_oserror(e)).startswith("torch_cuda_linalg"):
                import ctypes

                from torch.utils.cpp_extension import TORCH_LIB_PATH

                ctypes.CDLL(os.path.join(TORCH_LIB_PATH, missing_lib))
                torch.ops.load_library(lib_path)
            else:
                raise

        _HAS_EXT = True

        # Check torch version vs. compilation version
        # Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
        # https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/__init__.py
        flk_cuda_version = torch.ops.falkon._cuda_version()
        if torch.cuda.is_available() and flk_cuda_version != -1:
            if flk_cuda_version < 10000:
                f_major, f_minor = int(str(flk_cuda_version)[0]), int(str(flk_cuda_version)[2])
            else:
                f_major, f_minor = int(str(flk_cuda_version)[0:2]), int(str(flk_cuda_version)[3])
            t_major, t_minor = (int(x) for x in torch.version.cuda.split("."))

            if t_major != f_major:
                raise RuntimeError(
                    f"PyTorch and Falkon were compiled with different CUDA versions. "
                    f"PyTorch has CUDA version {t_major}.{t_minor} and Falkon has CUDA version "
                    f"{f_major}.{f_minor}. Please reinstall Falkon such that its version matches "
                    f"your PyTorch install."
                )
except ImportError:
    # if failed, try with JIT compilation
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    pt_version = torch_version()
    sources = (
        glob.glob(osp.join(ext_dir, "ops", "cpu", "*.cpp"))
        + glob.glob(osp.join(ext_dir, "ops", "autograd", "*.cpp"))
        + glob.glob(osp.join(ext_dir, "ops", "*.cpp"))
        + glob.glob(osp.join(ext_dir, "*.cpp"))
    )
    extra_cflags = [
        "-O3",
        f"-DTORCH_VERSION_MAJOR={pt_version[0]}",
        f"-DTORCH_VERSION_MINOR={pt_version[1]}",
        f"-DTORCH_VERSION_PATCH={pt_version[2]}",
    ]
    extra_ldflags = []
    extra_include_paths = []
    extra_cuda_cflags = []
    if cuda_toolkit_available():
        sources.extend(glob.glob(osp.join(ext_dir, "ops", "cuda", "*.cu")))
        extra_cflags += ["-DWITH_CUDA=1"]
        extra_cuda_cflags += ["--expt-relaxed-constexpr", "--extended-lambda"]
        from torch.utils.cpp_extension import CUDA_HOME, TORCH_LIB_PATH

        extra_ldflags += [
            "-L",
            os.path.join(CUDA_HOME, "lib"),
            "-L",
            TORCH_LIB_PATH,
            "-Wl,-rpath",
            TORCH_LIB_PATH,
            "-l",
            "cusparse",
            "-l",
            "cublas",
            "-l",
            "cusolver",
        ]
        if torch.__version__ >= (1, 12):
            extra_ldflags.extend(["-l", "torch_cuda_linalg"])
    else:
        warnings.warn("No CUDA toolkit found. Falkon will only run on the CPU.")

    name = "falkon.c_ext._C"
    build_dir = _get_build_directory(name, verbose=False)
    sources = sorted(sources)
    if len(os.listdir(build_dir)) == 0:
        # If the build exists, we assume the extension has been built and we can load it.
        # Otherwise we must build from scratch.
        # Remove the build directory just to be safe: pytorch jit might stuck if the build
        # directory exists.
        shutil.rmtree(build_dir)
        print("Building C extension. This might take a couple of minutes.")
    load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        is_python_module=False,
        is_standalone=False,
    )
    _HAS_EXT = True


def _assert_has_ext():
    if not _HAS_EXT:
        raise RuntimeError(
            "Couldn't load custom C++ ops. This can happen if your PyTorch and "
            "falkon versions are incompatible, or if you had errors while compiling "
            "falkon from source. For further information on the compatible versions, check "
            "https://github.com/falkonml/falkon#installation for the compatibility matrix. "
            "Please check your PyTorch version with torch.__version__ and your falkon "
            "version with falkon.__version__ and verify if they are compatible, and if not "
            "please reinstall falkon so that it matches your PyTorch install."
        )
