import warnings
from typing import Optional

import torch

from falkon.options import BaseOptions, KeopsOptions


def decide_cuda(opt: Optional[BaseOptions] = None):
    if opt is None:
        opt = BaseOptions()

    if opt.use_cpu:
        return False

    def get_error_str(name, err):
        e_str = (
            "Failed to initialize %s library; "
            "falling back to CPU. Set 'use_cpu' to "
            "True to avoid this warning." % (name)
        )
        if err is not None:
            e_str += "\nError encountered was %s" % (err)
        return e_str

    if not torch.cuda.is_available():
        warnings.warn(get_error_str("CUDA", None))
        return False
    return True


def decide_keops(opt: Optional[KeopsOptions] = None):
    if opt is None:
        opt = KeopsOptions()
    if opt.keops_active.lower() == "no":
        return False
    if opt.keops_active.lower() == "force":
        return True
    # If not 'no' or 'force' we can choose depending on whether keops works.
    if not hasattr(decide_keops, "keops_works"):
        try:
            import pykeops  # noqa: F401

            # pykeops.clean_pykeops()          # just in case old build files are still present
            # pykeops.test_torch_bindings()
            decide_keops.keops_works = True
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                "Failed to import PyKeops library; this might lead to "
                "slower matrix-vector products within FALKON. Please "
                "install PyKeops and check it works to suppress this warning."
            )
            decide_keops.keops_works = False
    return decide_keops.keops_works
