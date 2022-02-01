import os
# Set library exports
from . import kernels, sparse, center_selection, preconditioner, optim, gsc_losses, hopt
from .options import FalkonOptions
from .models import Falkon, LogisticFalkon, InCoreFalkon


init_dir = os.path.dirname(os.path.abspath(__file__))
# Set __version__ attribute on the package
with open(os.path.join(init_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()


__all__ = (
    'Falkon',
    'LogisticFalkon',
    'InCoreFalkon',
    'kernels',
    'optim',
    'preconditioner',
    'center_selection',
    'sparse',
    'gsc_losses',
    'hopt',
    'FalkonOptions',
)
