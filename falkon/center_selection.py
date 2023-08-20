import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch

from falkon.sparse import SparseTensor
from falkon.utils import check_random_generator
from falkon.utils.tensor_helpers import create_same_stride

__all__ = ("CenterSelector", "FixedSelector", "UniformSelector")
_tensor_type = Union[torch.Tensor, SparseTensor]
_opt_tns_tup = Union[_tensor_type, Tuple[_tensor_type, torch.Tensor]]
_opt_tns_idx_tup = Union[Tuple[_tensor_type, torch.Tensor], Tuple[_tensor_type, torch.Tensor, torch.Tensor]]


class CenterSelector(ABC):
    """Create the center selector with a random number generator

    Parameters
    ----------
    random_gen
        A numpy random number generator object or a random seed.
    """

    def __init__(self, random_gen):
        self.random_gen = check_random_generator(random_gen)

    @abstractmethod
    def select(self, X, Y) -> _opt_tns_tup:
        """Abstract method for selecting `M` centers from the data.

        Parameters
        ----------
        X
            The full input dataset (or a representation of it)
        Y
            The full input labels (this may be None)

        Returns
        -------
        X_centers
            If Y is None this is the only output: M centers selected from the data.
        Y_centers
            If Y is not empty, a set of label centers shall be returned as well.
        """
        pass

    @abstractmethod
    def select_indices(self, X, Y) -> _opt_tns_idx_tup:
        """Abstract method for selecting `M` centers from the data.

        Parameters
        ----------
        X
            The full input dataset (or a representation of it)
        Y
            The full input labels (this may be None)

        Returns
        -------
        X_centers
            If Y is None this is the only output: M centers selected from the data.
        Y_centers
            If Y is not empty, a set of label centers shall be returned as well.
        indices
            The indices in X associated with the chosen centers. Subclasses may not implement
            this method, in which case a `NotImplementedError` will be raised.
        """
        pass


class FixedSelector(CenterSelector):
    """Center selector which always picks the same centers.

    The fixed centers are specified at class initialization time.

    Parameters
    ----------
    centers
        Tensor of data-centers to be used.
    y_centers
        Optional tensor of label-centers to be used. If this is `None`, calling :meth:`select` with
        a non-empty `Y` argument will throw an exception
    idx_centers
        Optional tensor containing the indices which correspond to the given centers. This tensor
        is used in the :meth:`select_indices` method.
    """

    def __init__(
        self,
        centers: _tensor_type,
        y_centers: Optional[torch.Tensor] = None,
        idx_centers: Optional[torch.Tensor] = None,
    ):
        super().__init__(random_gen=None)
        self.centers = centers
        self.idx_centers = idx_centers
        self.y_centers = y_centers

    def select(self, X: _tensor_type, Y: Optional[torch.Tensor]) -> _opt_tns_tup:
        """Returns the fixed centers with which this instance was created

        Parameters
        ----------
        X
            This parameter is ignored. The centers returned are the ones passed in the class's
            constructor.
        Y
            Optional N x T tensor containing the input targets. The value of the parameter is
            ignored, but if it is not `None`, this method will return a tuple of X-centers
            and Y-centers.

        Returns
        -------
        X_M
            The fixed centers as given in the class constructor
        (X_M, Y_M)
            The X-centers and Y-centers as given in the class constructor. This tuple is only
            returned if `Y` is not None.

        Raises
        ------
        RuntimeError
            If parameter `Y` is not None but the `y_centers` tensor passed to the class constructor
            is `None`.
        """
        if Y is not None:
            if self.y_centers is None:
                raise RuntimeError("FixedSelector has no y-centers available, but `Y` is not None.")
            return self.centers, self.y_centers
        return self.centers

    def select_indices(self, X: _tensor_type, Y: Optional[torch.Tensor]) -> _opt_tns_idx_tup:
        """Returns the fixed centers, and their indices with which this instance was created

        Parameters
        ----------
        X
            This parameter is ignored. The centers returned are the ones passed in the class's
            constructor.
        Y
            Optional N x T tensor containing the input targets. The value of the parameter is
            ignored, but if it is not `None`, this method will return a tuple of X-centers
            Y-centers, and indices.

        Returns
        -------
        (X_M, indices)
            The fixed centers and the indices as given in the class constructor
        (X_M, Y_M, indices)
            The X-centers, Y-centers and indices as given in the class constructor.
            This tuple is only returned if `Y` is not None.

        Raises
        ------
        RuntimeError
            If the indices passed to the class constructor are `None`.
        RuntimeError
            If parameter `Y` is not None but the `y_centers` tensor passed to the class constructor
            is `None`.
        """
        if self.idx_centers is None:
            raise RuntimeError("`select_indices` method called but no indices were provided.")
        if Y is not None:
            if self.y_centers is None:
                raise RuntimeError("FixedSelector has no y-centers available, but `Y` is not None.")
            return self.centers, self.y_centers, self.idx_centers
        return self.centers, self.idx_centers


class UniformSelector(CenterSelector):
    """Center selector which chooses from the full dataset uniformly at random (without replacement)

    Parameters
    ----------
    random_gen
        A numpy random number generator object or a random seed.
    num_centers
        The number of centers which should be selected by this class.
    """

    def __init__(self, random_gen, num_centers: int):
        self.num_centers = num_centers
        super().__init__(random_gen)

    def select_indices(self, X: _tensor_type, Y: Optional[torch.Tensor]) -> _opt_tns_idx_tup:
        """Select M observations from 2D tensor `X`, preserving device and memory order.

        The selection strategy is uniformly at random. To control the randomness,
        pass an appropriate numpy random generator to this class's constructor.

        This method behaves the same as :meth:`select` but additionally returns a `LongTensor`
        containing the indices of the chosen points.

        Parameters
        ----------
        X
            N x D tensor containing the whole input dataset. If N is lower than the number of
            centers this class is programmed to pick, a warning will be raised and only N centers
            will be returned.
        Y
            Optional N x T tensor containing the input targets. If `Y` is provided,
            the same observations selected for `X` will also be selected from `Y`.
            Certain models (such as :class:`falkon.models.LogisticFalkon`) require centers to be
            extracted from both predictors and targets, while others (such as
            :class:`falkon.models.Falkon`) only require the centers from the predictors.

        Returns
        -------
        (X_M, indices)
            The randomly selected centers and the corresponding indices.
            The centers will be stored in a new, memory-contiguous tensor and all
            characteristics of the input tensor will be preserved.
        (X_M, Y_M, indices)
            If parameter`Y` is not `None` then the entries of `Y` corresponding to the
            selected centers of `X` will also be returned.
        """
        N = X.shape[0]
        num_centers = self.num_centers
        if num_centers > N:
            warnings.warn(
                "Number of centers M greater than the " f"number of data-points. Setting `num_centers` to {N}",
                stacklevel=2,
            )
            num_centers = N
        idx = self.random_gen.choice(N, size=num_centers, replace=False)

        if isinstance(X, SparseTensor):
            X_sp = X.to_scipy()
            centers = X_sp[idx, :].copy()
            Xc = SparseTensor.from_scipy(centers)
            th_idx = torch.from_numpy(idx.astype(np.int64)).to(X.device)
        else:
            Xc = create_same_stride(
                (num_centers, X.shape[1]), other=X, dtype=X.dtype, device=X.device, pin_memory=False
            )
            th_idx = torch.from_numpy(idx.astype(np.int64)).to(X.device)
            torch.index_select(X, dim=0, index=th_idx, out=Xc)

        if Y is not None:
            Yc = create_same_stride(
                (num_centers, Y.shape[1]), other=Y, dtype=Y.dtype, device=Y.device, pin_memory=False
            )
            th_idx = torch.from_numpy(idx.astype(np.int64)).to(Y.device)
            torch.index_select(Y, dim=0, index=th_idx, out=Yc)
            return Xc, Yc, th_idx
        return Xc, th_idx

    def select(self, X: _tensor_type, Y: Optional[torch.Tensor]) -> _opt_tns_tup:
        """Select M observations from 2D tensor `X`, preserving device and memory order.

        The selection strategy is uniformly at random. To control the randomness,
        pass an appropriate numpy random generator to this class's constructor.

        Parameters
        ----------
        X
            N x D tensor containing the whole input dataset. If N is lower than the number of
            centers this class is programmed to pick, a warning will be raised and only N centers
            will be returned.
        Y
            Optional N x T tensor containing the input targets. If `Y` is provided,
            the same observations selected for `X` will also be selected from `Y`.
            Certain models (such as :class:`falkon.models.LogisticFalkon`) require centers to be
            extracted from both predictors and targets, while others (such as
            :class:`falkon.models.Falkon`) only require the centers from the predictors.

        Returns
        -------
        X_M
            The randomly selected centers. They will be in a new, memory-contiguous tensor.
            All characteristics of the input tensor will be preserved.
        (X_M, Y_M)
            If `Y` was different than `None` then the entries of `Y` corresponding to the
            selected centers of `X` will also be returned.
        """
        out = self.select_indices(X, Y)
        if len(out) == 2:
            return out[0]
        return out[0], out[1]
