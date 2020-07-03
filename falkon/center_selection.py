import warnings
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import torch

from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils.tensor_helpers import create_same_stride

__all__ = ("CenterSelector", "UniformSelector")
_tensor_type = Union[torch.Tensor, SparseTensor]


class CenterSelector(ABC):
    def __init__(self, random_gen):
        self.random_gen = random_gen

    @abstractmethod
    def select(self, X, Y, M):
        pass


class UniformSelector(CenterSelector):
    def __init__(self, random_gen):
        super().__init__(random_gen)

    def select(self,
               X: _tensor_type,
               Y: Union[torch.Tensor, None],
               M: int) -> Union[_tensor_type, Tuple[_tensor_type, torch.Tensor]]:
        """Select M observations from 2D tensor `X`, preserving device and memory order.

        The selection strategy is uniformly at random. To control the randomness,
        pass an appropriate numpy random generator to this class's constructor.

        Parameters
        ----------
        X
            N x D tensor containing the whole input dataset. We have that N <= M.
        Y
            Optional N x T tensor containing the input targets. If `Y` is provided,
            the same observations selected for `X` will also be selected from `Y`.
            Certain models (such as :class:`falkon.LogisticFalkon`) require centers to be
            extracted from both predictors and targets, while others (such as
            :class:`falkon.Falkon`) only require the centers from the predictors.
        M
            The number of observations to choose. M <= N, otherwise M is forcibly set to N
            with a warning.

        Returns
        -------
        X_M
            The randomly selected centers. They will be in a new, memory-contiguous tensor.
            All characteristics of the input tensor will be preserved.
        (X_M, Y_M)
            If `Y` was different than `None` then the entries of `Y` corresponding to the
            selected centers of `X` will also be returned.
        """
        N = X.shape[0]
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
            Xc = create_same_stride((M, X.shape[1]), other=X, dtype=X.dtype, device=X.device,
                                    pin_memory=False)
            th_idx = torch.from_numpy(idx.astype(np.long)).to(X.device)
            torch.index_select(X, dim=0, index=th_idx, out=Xc)

        if Y is not None:
            Yc = create_same_stride((M, Y.shape[1]), other=Y, dtype=Y.dtype, device=Y.device,
                                    pin_memory=False)
            th_idx = torch.from_numpy(idx.astype(np.long)).to(Y.device)
            torch.index_select(Y, dim=0, index=th_idx, out=Yc)
            return Xc, Yc
        return Xc
