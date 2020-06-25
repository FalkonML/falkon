import os

from . import kernels, optim, preconditioner, center_selection
from .models import Falkon, LogisticFalkon
from .options import FalkonOptions

__all__ = ('Falkon', 'LogisticFalkon', 'kernels', 'optim', 'preconditioner', 'center_selection',
           'FalkonOptions')


# Set __version__ attribute on the package
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

