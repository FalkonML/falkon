from abc import abstractmethod
from typing import Union, Tuple

import h5py
import numpy as np
import scipy.io as scio
import scipy.sparse

from benchmark_utils import Dataset

__all__ = (
    "get_load_fn", "get_cv_fn",
    "BaseDataset", "HiggsDataset", "SusyDataset", "MillionSongsDataset",
    "TimitDataset", "NycTaxiDataset", "YelpDataset", "FlightsDataset"
)


def standardize_x(Xtr, Xts):
    if isinstance(Xtr, np.ndarray):
        mXtr = Xtr.mean(axis=0, keepdims=True, dtype=np.float64).astype(Xtr.dtype)
        sXtr = Xtr.std(axis=0, keepdims=True, dtype=np.float64, ddof=1).astype(Xtr.dtype)
    else:
        mXtr = Xtr.mean(dim=0, keepdims=True)
        sXtr = Xtr.std(dim=0, keepdims=True)

    Xtr -= mXtr
    Xtr /= sXtr
    Xts -= mXtr
    Xts /= sXtr

    return Xtr, Xts, {}


def as_np_dtype(dtype):
    if "float32" in str(dtype):
        return np.float32
    if "float64" in str(dtype):
        return np.float64
    if "int32" in str(dtype):
        return np.int32
    raise ValueError(dtype)


def as_torch_dtype(dtype):
    import torch
    if "float32" in str(dtype):
        return torch.float32
    if "float64" in str(dtype):
        return torch.float64
    if "int32" in str(dtype):
        return torch.int32
    raise ValueError(dtype)


def equal_split(N, train_frac):
    Ntr = int(N * train_frac)
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_tr = idx[:Ntr]
    idx_ts = idx[Ntr:]
    return idx_tr, idx_ts


class MyKFold():
    def __init__(self, n_splits, shuffle, seed=92):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed)

    def split(self, X, y=None):
        N = X.shape[0]
        indices = np.arange(N)
        mask = np.full(N, False)
        if self.shuffle:
            self.random_state.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, N // n_splits, dtype=np.int)
        fold_sizes[:N % n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            mask.fill(False)
            mask[indices[start:stop]] = True
            yield mask
            current = stop


class BaseDataset():
    def load_data(self, dtype, as_torch=False, as_tf=False):
        X, Y = self.read_data(dtype)
        print(f"Loaded {self.dset_name()} dataset in {dtype} precision.", flush=True)
        Xtr, Ytr, Xts, Yts = self.split_data(X, Y, train_frac=None)
        assert Xtr.shape[0] == Ytr.shape[0]
        assert Xts.shape[0] == Yts.shape[0]
        assert Xtr.shape[1] == Xts.shape[1]
        print(f"Split the data into {Xtr.shape[0]} training, "
              f"{Xts.shape[0]} validation points of dimension {Xtr.shape[1]}.", flush=True)
        Xtr, Xts, other_X = self.preprocess_x(Xtr, Xts)
        Ytr, Yts, other_Y = self.preprocess_y(Ytr, Yts)
        print("Data-preprocessing completed.", flush=True)
        kwargs = dict()
        kwargs.update(other_X)
        kwargs.update(other_Y)
        if as_torch:
            return self.to_torch(Xtr, Ytr, Xts, Yts, **kwargs)
        if as_tf:
            return self.to_tensorflow(Xtr, Ytr, Xts, Yts, **kwargs)
        return Xtr, Ytr, Xts, Yts, kwargs

    def load_data_cv(self, dtype, k, as_torch=False):
        X, Y = self.read_data(dtype)
        print(f"Loaded {self.dset_name()} dataset in {dtype} precision.", flush=True)
        print(f"Data size: {X.shape[0]} points with {X.shape[1]} features", flush=True)

        kfold = MyKFold(n_splits=k, shuffle=True)
        iteration = 0
        for test_idx in kfold.split(X):
            Xtr = X[~test_idx]
            Ytr = Y[~test_idx]
            Xts = X[test_idx]
            Yts = Y[test_idx]
            Xtr, Xts, other_X = self.preprocess_x(Xtr, Xts)
            Ytr, Yts, other_Y = self.preprocess_y(Ytr, Yts)
            print("Preprocessing complete (iter %d) - Divided into %d train, %d test points" %
                  (iteration, Xtr.shape[0], Xts.shape[0]))
            kwargs = dict()
            kwargs.update(other_X)
            kwargs.update(other_Y)
            if as_torch:
                yield self.to_torch(Xtr, Ytr, Xts, Yts, **kwargs)
            else:
                yield Xtr, Ytr, Xts, Yts, kwargs
            iteration += 1

    @staticmethod
    @abstractmethod
    def read_data(dtype):
        pass

    @staticmethod
    @abstractmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        pass

    @staticmethod
    @abstractmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    @abstractmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr, Yts, {}

    @staticmethod
    def to_torch(Xtr, Ytr, Xts, Yts, **kwargs):
        import torch
        #torch_kwargs = {k: torch.from_numpy(v) for k, v in kwargs.items()}
        torch_kwargs = kwargs
        return (
            torch.from_numpy(Xtr),
            torch.from_numpy(Ytr),
            torch.from_numpy(Xts),
            torch.from_numpy(Yts),
            torch_kwargs
        )

    @staticmethod
    def to_tensorflow(Xtr, Ytr, Xts, Yts, **kwargs):
        # By default tensorflow is happy with numpy arrays
        return (Xtr, Ytr, Xts, Yts, kwargs)

    @abstractmethod
    def dset_name(self) -> str:
        return "UNKOWN"


class MillionSongsDataset(BaseDataset):
    file_name = '/data/DATASETS/MillionSongs/YearPredictionMSD.mat'
    _dset_name = 'MillionSongs'

    @staticmethod
    def read_data(dtype) -> Tuple[np.ndarray, np.ndarray]:
        f = scio.loadmat(MillionSongsDataset.file_name)
        X = f['X'][:, 1:].astype(as_np_dtype(dtype))
        Y = f['X'][:, 0].astype(as_np_dtype(dtype))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac=None):
        if train_frac == 'auto' or train_frac is None:
            idx_tr = np.arange(463715)
            idx_ts = np.arange(463715, 463715 + 51630)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)

        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
        sttr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
        Ytr -= mtr
        Ytr /= sttr
        Yts -= mtr
        Yts /= sttr
        Ytr = Ytr.reshape((-1, 1))
        Yts = Yts.reshape((-1, 1))
        return Ytr, Yts, {'Y_std': sttr, 'Y_mean': mtr}

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def dset_name(self):
        return self._dset_name


class NycTaxiDataset(BaseDataset):
    file_name = '/data/DATASETS/NYCTAXI/NYCTAXI.h5'
    _dset_name = 'TAXI'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(NycTaxiDataset.file_name, 'r')
        X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))  # N x 9
        Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))  # N x 1

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = NycTaxiDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.std(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr

        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
        sttr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
        Ytr -= mtr
        Ytr /= sttr
        Yts -= mtr
        Yts /= sttr
        return Ytr, Yts, {'Y_std': sttr}

    def dset_name(self):
        return self._dset_name


class HiggsDataset(BaseDataset):
    file_name = '/data/DATASETS/HIGGS_UCI/Higgs.mat'
    _dset_name = 'HIGGS'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(HiggsDataset.file_name, 'r')
        arr = np.array(h5py_file['X'], dtype=as_np_dtype(dtype)).T
        X = arr[:, 1:]
        Y = arr[:, 0]
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = HiggsDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.var(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr

        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class TimitDataset(BaseDataset):
    file_name = '/data/DATASETS/TIMIT/TIMIT.mat'
    _dset_name = 'TIMIT'

    @staticmethod
    def read_data(dtype):
        f = scio.loadmat(TimitDataset.file_name)
        dtype = as_np_dtype(dtype)
        Xtr = np.array(f['Xtr'], dtype=dtype)
        Xts = np.array(f['Xts'], dtype=dtype)
        Ytr = np.array(f['Ytr'], dtype=dtype).reshape((-1, ))
        Yts = np.array(f['Yts'], dtype=dtype).reshape((-1, ))
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            # Default split recovers the original Xtr, Xts split
            idx_tr = np.arange(1124823)
            idx_ts = np.arange(1124823, 1124823 + 57242)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)

        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 144
        damping = 1 / (n_classes - 1)
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye - damping + eye * damping
        # Ytr
        Ytr = A[Ytr.astype(np.int32), :]
        # Yts
        Yts = (Yts - 1) * 3
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class YelpDataset(BaseDataset):
    file_name = '/data/DATASETS/YELP_Ben/YELP_Ben_OnlyONES.mat'
    _dset_name = 'YELP'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        dtype = as_np_dtype(dtype)
        f = h5py.File(YelpDataset.file_name, 'r')
        X = scipy.sparse.csc_matrix((
            np.array(f['X']['data'], dtype),
            f['X']['ir'][...], f['X']['jc'][...])).tocsr(copy=False)
        Y = np.array(f['Y'], dtype=dtype).reshape((-1, 1))
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = YelpDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # scaler = sklearn.preprocessing.StandardScaler(copy=False, with_mean=False, with_std=True)
        # Xtr = scaler.fit_transform(Xtr)
        # Xts = scaler.transform(Xts)
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr, Yts, {}

    @staticmethod
    def to_torch(Xtr, Ytr, Xts, Yts, **kwargs):
        from falkon.sparse.sparse_tensor import SparseTensor
        import torch
        return (SparseTensor.from_scipy(Xtr),
                torch.from_numpy(Ytr),
                SparseTensor.from_scipy(Xts),
                torch.from_numpy(Yts), {})

    @staticmethod
    def to_tensorflow(Xtr, Ytr, Xts, Yts, **kwargs):
        import tensorflow as tf
        def scipy2tf(X):
            # Uses same representation as pytorch
            # https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
            coo = X.tocoo()
            indices = np.array([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)
        return (scipy2tf(Xtr),
                Ytr,
                scipy2tf(Xts),
                Yts,
                {})

    def dset_name(self):
        return self._dset_name


class FlightsDataset(BaseDataset):
    file_name = '/data/DATASETS/FLIGHTS/flights.hdf5'
    _dset_name = 'FLIGHTS'
    _default_train_frac = 0.666

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(FlightsDataset.file_name, 'r')
        X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
        Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        # Preprocessing independent of train/test
        # As for https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_additive_figure.py
        # 1. Convert time of day from hhmm to minutes since midnight
        #  ArrTime is column 7, DepTime is column 6
        X[:,7] = 60*np.floor(X[:,7]/100) + np.mod(X[:,7], 100)
        X[:,6] = 60*np.floor(X[:,6]/100) + np.mod(X[:,6], 100)
        # 2. remove flights with silly negative delays (small negative delays are OK)
        pos_delay_idx = np.where(Y > -60)[0]
        X = X[pos_delay_idx, :]
        Y = Y[pos_delay_idx, :]
        # 3. remove outlying flights in term of length (col 'AirTime' at pos 5)
        short_flight_idx = np.where(X[:,5] < 700)[0]
        X = X[short_flight_idx, :]
        Y = Y[short_flight_idx, :]

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = FlightsDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
        sttr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
        Ytr -= mtr
        Ytr /= sttr
        Yts -= mtr
        Yts /= sttr
        Ytr = Ytr.reshape((-1, 1))
        Yts = Yts.reshape((-1, 1))
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class FlightsClsDataset(BaseDataset):
    file_name = '/data/DATASETS/FLIGHTS/flights.hdf5'
    _dset_name = 'FLIGHTS-CLS'
    _default_train_num = 100_000

    @staticmethod
    def read_data(dtype):
        h5py_file = h5py.File(FlightsDataset.file_name, 'r')
        X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
        Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
        # Preprocessing independent of train/test
        # As for https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_additive_figure.py
        # 1. Convert time of day from hhmm to minutes since midnight
        #  ArrTime is column 7, DepTime is column 6
        X[:,7] = 60*np.floor(X[:,7]/100) + np.mod(X[:,7], 100)
        X[:,6] = 60*np.floor(X[:,6]/100) + np.mod(X[:,6], 100)
        # Turn regression into classification by thresholding delay or not delay:
        Y = (Y <= 0).astype(X.dtype)

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = (X.shape[0] - FlightsClsDataset._default_train_num) / X.shape[0]
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name


class SusyDataset(BaseDataset):
    file_name = '/data/DATASETS/SUSY/Susy.mat'
    _dset_name = 'SUSY'
    _default_train_frac = 0.8

    @staticmethod
    def read_data(dtype):
        with h5py.File(SusyDataset.file_name, "r") as f:
            arr = np.asarray(f['X'], dtype=as_np_dtype(dtype)).T
            X = arr[:, 1:]
            Y = arr[:, 0].reshape(-1, 1)
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = SusyDataset._default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Convert labels from 0, 1 to -1, +1"""
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}

    def dset_name(self):
        return self._dset_name

class CIFAR10Dataset(BaseDataset):
    file_name = "/data/DATASETS/CIFAR10/cifar10.mat"
    ts_file_name = "/data/DATASETS/CIFAR10/cifar10.t.mat"
    _dset_name = "CIFAR10"

    @staticmethod
    def read_data(dtype):
        # Read Training data
        data = scio.loadmat(CIFAR10Dataset.file_name)
        Xtr = data["Z"].astype(as_np_dtype(dtype)) / 255
        Ytr = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Read Testing data
        data = scio.loadmat(CIFAR10Dataset.ts_file_name)
        Xts = data["Z"].astype(as_np_dtype(dtype)) / 255
        Yts = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Merge
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)
        # Convert to RGB
        R = X[:, :1024]
        G = X[:, 1024:2048]
        B = X[:, 2048:3072]
        X = 0.2126 * R + 0.7152 * G + 0.0722 * B
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac):
        if train_frac is None:
            idx_tr = np.arange(0, 50000)
            idx_ts = np.arange(50000, 60000)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye
        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name



class SVHNDataset(BaseDataset):
    file_name = "/data/DATASETS/SVHN/SVHN.mat"
    ts_file_name = "/data/DATASETS/SVHN/SVHN.t.mat"
    _dset_name = "SVHN"

    @staticmethod
    def read_data(dtype):
        # Read Training data
        data = scio.loadmat(SVHNDataset.file_name)
        Xtr = data["Z"].astype(as_np_dtype(dtype)) / 255
        Ytr = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Read Testing data
        data = scio.loadmat(SVHNDataset.ts_file_name)
        Xts = data["Z"].astype(as_np_dtype(dtype)) / 255
        Yts = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))
        # Merge
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)
        # Convert to RGB
        R = X[:, :1024]
        G = X[:, 1024:2048]
        B = X[:, 2048:3072]
        X = 0.2126 * R + 0.7152 * G + 0.0722 * B
        # Y -- for some reason it's 1 indexed
        Y = Y - 1
        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac):
        if train_frac is None:
            idx_tr = np.arange(0, 73257)
            idx_ts = np.arange(73257, 73257 + 26032)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye
        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class MnistSmallDataset(BaseDataset):
    file_name = "/data/DATASETS/MNIST/mnist.mat"
    ts_file_name = "/data/DATASETS/MNIST/mnist.t.mat"
    _dset_name = "MNIST"

    @staticmethod
    def read_data(dtype):
        data = scio.loadmat(MnistSmallDataset.file_name)
        Xtr = data["Z"].astype(as_np_dtype(dtype)) / 255
        Ytr = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))

        data = scio.loadmat(MnistSmallDataset.ts_file_name)
        Xts = data["Z"].astype(as_np_dtype(dtype)) / 255
        # For no reason MNIST has 778 features here.. Add zeros at end?
        Xts = np.concatenate((np.zeros((Xts.shape[0], 2), dtype=Xts.dtype), Xts), axis=1)
        Yts = data["y"].astype(as_np_dtype(dtype)).reshape((-1, ))

        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)

        return X, Y

    @staticmethod
    def split_data(X, Y, train_frac):
        if train_frac is None:
            idx_tr = np.arange(0, 60000)
            idx_ts = np.arange(60000, 70000)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye
        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]
        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


class MnistDataset(BaseDataset):
    file_name = '/data/DATASETS/MNIST/mnist8m_normalized.hdf5'
    _dset_name = 'MNIST8M'
    num_train = 6750000
    num_test = 10_000

    @staticmethod
    def read_data(dtype):
        with h5py.File(MnistDataset.file_name, "r") as f:
            Xtr = np.array(f["X_train"], dtype=as_np_dtype(dtype))
            Ytr = np.array(f["Y_train"], dtype=as_np_dtype(dtype))
            Xts = np.array(f["X_test"], dtype=as_np_dtype(dtype))
            Yts = np.array(f["Y_test"], dtype=as_np_dtype(dtype))
        return np.concatenate((Xtr, Xts), 0), np.concatenate((Ytr, Yts), 0)

    @staticmethod
    def split_data(X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            idx_tr = np.arange(MnistDataset.num_train)
            idx_ts = np.arange(MnistDataset.num_train, MnistDataset.num_train + MnistDataset.num_test)
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @staticmethod
    def preprocess_x(Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @staticmethod
    def preprocess_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        n_classes = 10
        damping = 1 / (n_classes)
        eye = np.eye(n_classes, dtype=as_np_dtype(Ytr.dtype))
        A = eye - damping #+ eye * damping

        Ytr = A[Ytr.astype(np.int32), :]
        Yts = A[Yts.astype(np.int32), :]

        return Ytr, Yts, {}

    def dset_name(self):
        return self._dset_name


""" Public API """

__LOADERS = {
    Dataset.TIMIT: TimitDataset(),
    Dataset.HIGGS: HiggsDataset(),
    Dataset.MILLIONSONGS: MillionSongsDataset(),
    Dataset.TAXI: NycTaxiDataset(),
    Dataset.YELP: YelpDataset(),
    Dataset.FLIGHTS: FlightsDataset(),
    Dataset.SUSY: SusyDataset(),
    Dataset.MNIST: MnistDataset(),
    Dataset.FLIGHTS_CLS: FlightsClsDataset(),
    Dataset.SVHN: SVHNDataset(),
    Dataset.MNIST_SMALL: MnistSmallDataset(),
    Dataset.CIFAR10: CIFAR10Dataset(),
}


def get_load_fn(dset: Dataset):
    try:
        return __LOADERS[dset].load_data
    except KeyError:
        raise KeyError(dset, f"No loader function found for dataset {dset}.")


def get_cv_fn(dset: Dataset):
    try:
        return __LOADERS[dset].load_data_cv
    except KeyError:
        raise KeyError(dset, f"No CV-loader function found for dataset {dset}.")
