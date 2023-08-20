import os

# Library exports
from . import center_selection, gsc_losses, hopt, kernels, optim, preconditioner, sparse
from .models import Falkon, InCoreFalkon, LogisticFalkon
from .options import FalkonOptions

# Set __version__ attribute on the package
init_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(init_dir, "VERSION")) as version_file:
    __version__ = version_file.read().strip()

__all__ = (
    "Falkon",
    "LogisticFalkon",
    "InCoreFalkon",
    "FalkonOptions",
    "kernels",
    "optim",
    "preconditioner",
    "center_selection",
    "sparse",
    "gsc_losses",
    "hopt",
    "__version__",
)
