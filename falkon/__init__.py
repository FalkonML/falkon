import os

from .options import FalkonOptions  # isort:skip
from . import (  # isort:skip
    center_selection,
    sparse,
    kernels,
    preconditioner,
    optim,
    gsc_losses,
    hopt,
)
from .models import Falkon, InCoreFalkon, LogisticFalkon  # isort:skip

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
