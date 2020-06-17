from typing import Union, Tuple
from abc import ABC, abstractmethod
import warnings

import torch
import numpy as np
from falkon.sparse.sparse_tensor import SparseTensor

from falkon.utils.tensor_helpers import is_f_contig
from falkon.utils import CompOpt


_tensor_type = Union[torch.Tensor, SparseTensor]


class NySel(ABC):
    def __init__(self, random_gen, opt=None):
        if opt is not None:
            self.params = CompOpt(opt)
        else:
            self.params = CompOpt()
        self.random_gen = random_gen

    @abstractmethod
    def select(self, X, Y, M):
        pass


class UniformSel(NySel):
    def __init__(self, random_gen, opt=None):
        super().__init__(random_gen, opt)

    def select(self,
               X: _tensor_type,
               Y: Union[torch.Tensor, None],
               M: int) -> Union[_tensor_type, Tuple[_tensor_type, torch.Tensor]]:
        """Select M rows from 2D array `X`, preserving the memory order of `X`.
        """
        N = X.size(0)
        if M > N:
            warnings.warn("Number of centers M greater than the "
                          "number of data-points. Setting M to %d" % (N))
            M = N
        idx = self.random_gen.choice(N, size=M, replace=False)

        if isinstance(X, SparseTensor):
            X = X.to_scipy()
            centers = X[idx, :].copy()
            Xc = SparseTensor.from_scipy(centers)
        else:
            Xnp = X.numpy()  # work on np array
            if is_f_contig(X):
                order = 'F'
            else:
                order = 'C'
            Xc_np = np.empty((M, Xnp.shape[1]), dtype=Xnp.dtype, order=order)
            Xc = torch.from_numpy(np.take(Xnp, idx, axis=0, out=Xc_np, mode='wrap'))

        if Y is not None:
            Ynp = Y.numpy()  # work on np array
            if is_f_contig(X):
                order = 'F'
            else:
                order = 'C'
            Yc_np = np.empty((M, Ynp.shape[1]), dtype=Ynp.dtype, order=order)
            Yc = torch.from_numpy(np.take(Ynp, idx, axis=0, out=Yc_np, mode='wrap'))
            return Xc, Yc
        return Xc
