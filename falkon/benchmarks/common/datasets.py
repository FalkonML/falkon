import os
from abc import abstractmethod, ABC
from typing import Union, Tuple

import h5py
import numpy as np
import scipy.io as scio
import scipy.sparse
from scipy.sparse import load_npz
from sklearn.datasets import load_svmlight_file

from .benchmark_utils import Dataset

__all__ = (
    "get_load_fn", "get_cv_fn",
    "BaseDataset", "HiggsDataset", "SusyDataset", "MillionSongsDataset",
    "TimitDataset", "NycTaxiDataset", "YelpDataset", "FlightsDataset"
)


def load_from_npz(dset_name, folder, dtype, verbose=False):
    x_file = os.path.join(folder, "%s_data.npz" % dset_name)
    y_file = os.path.join(folder, "%s_target.npy" % dset_name)
    x_data = np.asarray(load_npz(x_file).todense()).astype(as_np_dtype(dtype))
    y_data = np.load(y_file).astype(as_np_dtype(dtype))
    if verbose:
        print("Loaded %s. X: %s - Y: %s" % (dset_name, x_data.shape, y_data.shape))
    return (x_data, y_data)


def load_from_t(dset_name, folder, verbose=False):
    file_tr = os.path.join(folder, dset_name)
    file_ts = os.path.join(folder, dset_name + ".t")
    x_data_tr, y_data_tr = load_svmlight_file(file_tr)
    x_data_tr = np.asarray(x_data_tr.todense())
    x_data_ts, y_data_ts = load_svmlight_file(file_ts)
    x_data_ts = np.asarray(x_data_ts.todense())
    if verbose:
        print("Loaded %s. train X: %s - Y: %s - test X: %s - Y: %s" %
              (dset_name, x_data_tr.shape, y_data_tr.shape, x_data_ts.shape, y_data_ts.shape))
    x_data = np.concatenate((x_data_tr, x_data_ts))
    y_data = np.concatenate((y_data_tr, y_data_ts))
    return x_data, y_data


def standardize_x(Xtr, Xts):
    if isinstance(Xtr, np.ndarray):
        mXtr = Xtr.mean(axis=0, keepdims=True, dtype=np.float64).astype(Xtr.dtype)
        sXtr = Xtr.std(axis=0, keepdims=True, dtype=np.float64, ddof=1).astype(Xtr.dtype)
    else:
        mXtr = Xtr.mean(dim=0, keepdims=True)
        sXtr = Xtr.std(dim=0, keepdims=True)
    sXtr[sXtr == 0] = 1.0

    Xtr -= mXtr
    Xtr /= sXtr
    Xts -= mXtr
    Xts /= sXtr

    return Xtr, Xts, {}


def mean_remove_y(Ytr, Yts):
    mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
    Ytr -= mtr
    Yts -= mtr
    Ytr = Ytr.reshape((-1, 1))
    Yts = Yts.reshape((-1, 1))
    return Ytr, Yts, {'Y_mean': mtr}


def standardize_y(Ytr, Yts):
    mtr = np.mean(Ytr, dtype=np.float64).astype(Ytr.dtype)
    stdtr = np.std(Ytr, dtype=np.float64, ddof=1).astype(Ytr.dtype)
    Ytr -= mtr
    Ytr /= stdtr
    Yts -= mtr
    Yts /= stdtr
    Ytr = Ytr.reshape((-1, 1))
    Yts = Yts.reshape((-1, 1))
    return Ytr, Yts, {'Y_mean': mtr, 'Y_std': stdtr}


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


def convert_to_binary_y(Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    labels = set(np.unique(Ytr))
    if labels == {0, 1}:
        # Convert labels from 0, 1 to -1, +1
        Ytr = Ytr * 2 - 1
        Yts = Yts * 2 - 1
    elif labels == {1, 2}:
        # Convert from 1, 2 to -1, +1
        Ytr = (Ytr - 1) * 2 - 1
        Yts = (Yts - 1) * 2 - 1

    return Ytr.reshape(-1, 1), Yts.reshape(-1, 1), {}


def convert_to_onehot(Ytr: np.ndarray, Yts: np.ndarray, num_classes: int, damping: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
    eye = np.eye(num_classes, dtype=as_np_dtype(Ytr.dtype))
    if damping:
        damp_val = 1 / (num_classes - 1)
        eye = eye - damp_val  # + eye * damping
    Ytr = eye[Ytr.astype(np.int32).reshape(-1), :]
    Yts = eye[Yts.astype(np.int32).reshape(-1), :]
    return Ytr, Yts, {}


def rgb_to_bw(X, dim=32):
    img_len = dim**2
    R = X[:, :img_len]
    G = X[:, img_len:2 * img_len]
    B = X[:, 2 * img_len:3 * img_len]
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


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
        print(f"Loaded {self.dset_name} dataset in {dtype} precision.", flush=True)
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
        print(f"Loaded {self.dset_name} dataset in {dtype} precision.", flush=True)
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

    @abstractmethod
    def read_data(self, dtype):
        pass

    @abstractmethod
    def split_data(self, X, Y, train_frac: Union[float, None]):
        pass

    @abstractmethod
    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    @abstractmethod
    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr, Yts, {}

    def to_torch(self, Xtr, Ytr, Xts, Yts, **kwargs):
        import torch
        # torch_kwargs = {k: torch.from_numpy(v) for k, v in kwargs.items()}
        torch_kwargs = kwargs
        return (
            torch.from_numpy(Xtr),
            torch.from_numpy(Ytr),
            torch.from_numpy(Xts),
            torch.from_numpy(Yts),
            torch_kwargs
        )

    def to_tensorflow(self, Xtr, Ytr, Xts, Yts, **kwargs):
        # By default tensorflow is happy with numpy arrays
        return (Xtr, Ytr, Xts, Yts, kwargs)

    @property
    @abstractmethod
    def dset_name(self) -> str:
        pass


class KnownSplitDataset(BaseDataset, ABC):
    def split_data(self, X, Y, train_frac: Union[float, None, str] = None):
        if train_frac == 'auto' or train_frac is None:
            idx_tr = np.arange(self.num_train_samples)
            if self.num_test_samples > 0:
                idx_ts = np.arange(self.num_train_samples, self.num_train_samples + self.num_test_samples)
            else:
                idx_ts = np.arange(self.num_train_samples, X.shape[0])
        else:
            idx_tr, idx_ts = equal_split(X.shape[0], train_frac)

        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @property
    @abstractmethod
    def num_train_samples(self):
        pass

    @property
    def num_test_samples(self):
        return -1


class RandomSplitDataset(BaseDataset, ABC):
    def split_data(self, X, Y, train_frac: Union[float, None, str] = None):
        if train_frac is None:
            train_frac = self.default_train_frac
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    @property
    @abstractmethod
    def default_train_frac(self):
        pass


class Hdf5Dataset(BaseDataset, ABC):
    def read_data(self, dtype):
        with h5py.File(self.file_name, 'r') as h5py_file:
            if 'X_train' in h5py_file.keys() and 'X_test' in h5py_file.keys() and \
                    'Y_train' in h5py_file.keys() and 'Y_test' in h5py_file.keys():
                X_train = np.array(h5py_file['X_train'], dtype=as_np_dtype(dtype))
                Y_train = np.array(h5py_file['Y_train'], dtype=as_np_dtype(dtype))
                X_test = np.array(h5py_file['X_test'], dtype=as_np_dtype(dtype))
                Y_test = np.array(h5py_file['Y_test'], dtype=as_np_dtype(dtype))
                X = np.concatenate([X_train, X_test], axis=0)
                Y = np.concatenate([Y_train, Y_test], axis=0)
            elif 'X' in h5py_file.keys() and 'Y' in h5py_file.keys():
                X = np.array(h5py_file['X'], dtype=as_np_dtype(dtype))
                Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype))
            else:
                raise RuntimeError(f"Cannot parse h5py file with keys {list(h5py_file.keys())}")
        return X, Y

    @property
    @abstractmethod
    def file_name(self):
        pass


class MillionSongsDataset(KnownSplitDataset):
    file_name = '/data/DATASETS/MillionSongs/YearPredictionMSD.mat'
    dset_name = 'MillionSongs'
    num_train_samples = 463715
    num_test_samples = 51630

    def read_data(self, dtype) -> Tuple[np.ndarray, np.ndarray]:
        f = scio.loadmat(MillionSongsDataset.file_name)
        X = f['X'][:, 1:].astype(as_np_dtype(dtype))
        Y = f['X'][:, 0].astype(as_np_dtype(dtype))
        return X, Y

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)  # Original

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)


class NycTaxiDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/NYCTAXI/NYCTAXI.h5'
    dset_name = 'TAXI'
    default_train_frac = 0.8

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class HiggsDataset(RandomSplitDataset):
    file_name = '/data/DATASETS/HIGGS_UCI/Higgs.mat'
    dset_name = 'HIGGS'
    default_train_frac = 0.8

    def read_data(self, dtype):
        with h5py.File(HiggsDataset.file_name, 'r') as h5py_file:
            arr = np.array(h5py_file['X'], dtype=as_np_dtype(dtype)).T
        X = arr[:, 1:]
        Y = arr[:, 0]
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.var(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr

        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class TimitDataset(KnownSplitDataset):
    file_name = '/data/DATASETS/TIMIT/TIMIT.mat'
    dset_name = 'TIMIT'
    num_train_samples = 1124823

    def read_data(self, dtype):
        f = scio.loadmat(TimitDataset.file_name)
        dtype = as_np_dtype(dtype)
        Xtr = np.array(f['Xtr'], dtype=dtype)
        Xts = np.array(f['Xts'], dtype=dtype)
        Ytr = np.array(f['Ytr'], dtype=dtype).reshape((-1,))
        Yts = np.array(f['Yts'], dtype=dtype).reshape((-1,))
        X = np.concatenate((Xtr, Xts), axis=0)
        Y = np.concatenate((Ytr, Yts), axis=0)

        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        Yts = (Yts - 1) * 3
        return convert_to_onehot(Ytr, Yts, num_classes=144, damping=True)


class YelpDataset(RandomSplitDataset):
    file_name = '/data/DATASETS/YELP_Ben/YELP_Ben_OnlyONES.mat'
    dset_name = 'YELP'
    default_train_frac = 0.8

    def read_data(self, dtype):
        with h5py.File(YelpDataset.file_name, 'r') as h5py_file:
            X = scipy.sparse.csc_matrix((
                np.array(h5py_file['X']['data'], as_np_dtype(dtype)),
                h5py_file['X']['ir'][...], h5py_file['X']['jc'][...])).tocsr(copy=False)
            Y = np.array(h5py_file['Y'], dtype=as_np_dtype(dtype)).reshape((-1, 1))
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # scaler = sklearn.preprocessing.StandardScaler(copy=False, with_mean=False, with_std=True)
        # Xtr = scaler.fit_transform(Xtr)
        # Xts = scaler.transform(Xts)
        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr, Yts, {}

    def to_torch(self, Xtr, Ytr, Xts, Yts, **kwargs):
        from falkon.sparse.sparse_tensor import SparseTensor
        import torch
        return (SparseTensor.from_scipy(Xtr),
                torch.from_numpy(Ytr),
                SparseTensor.from_scipy(Xts),
                torch.from_numpy(Yts), {})

    def to_tensorflow(self, Xtr, Ytr, Xts, Yts, **kwargs):
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


class FlightsDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/FLIGHTS/flights.hdf5'
    dset_name = 'FLIGHTS'
    default_train_frac = 0.666

    def read_data(self, dtype):
        X, Y = super().read_data(dtype)
        # Preprocessing independent of train/test
        # As for https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_additive_figure.py
        # 1. Convert time of day from hhmm to minutes since midnight
        #  ArrTime is column 7, DepTime is column 6
        X[:, 7] = 60 * np.floor(X[:, 7] / 100) + np.mod(X[:, 7], 100)
        X[:, 6] = 60 * np.floor(X[:, 6] / 100) + np.mod(X[:, 6], 100)
        # 2. remove flights with silly negative delays (small negative delays are OK)
        pos_delay_idx = np.where(Y > -60)[0]
        X = X[pos_delay_idx, :]
        Y = Y[pos_delay_idx, :]
        # 3. remove outlying flights in term of length (col 'AirTime' at pos 5)
        short_flight_idx = np.where(X[:, 5] < 700)[0]
        X = X[short_flight_idx, :]
        Y = Y[short_flight_idx, :]

        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        Ytr, Yts, metadata = standardize_y(Ytr, Yts)
        return Ytr, Yts, {}


class FlightsClsDataset(Hdf5Dataset):
    file_name = '/data/DATASETS/FLIGHTS/flights.hdf5'
    dset_name = 'FLIGHTS-CLS'
    _default_train_num = 100_000

    def read_data(self, dtype):
        X, Y = super().read_data(dtype)
        # Preprocessing independent of train/test
        # As for https://github.com/jameshensman/VFF/blob/master/experiments/airline/airline_additive_figure.py
        # 1. Convert time of day from hhmm to minutes since midnight
        #  ArrTime is column 7, DepTime is column 6
        X[:, 7] = 60 * np.floor(X[:, 7] / 100) + np.mod(X[:, 7], 100)
        X[:, 6] = 60 * np.floor(X[:, 6] / 100) + np.mod(X[:, 6], 100)
        # Turn regression into classification by thresholding delay or not delay:
        Y = (Y <= 0).astype(X.dtype)

        return X, Y

    def split_data(self, X, Y, train_frac: Union[float, None]):
        if train_frac is None:
            train_frac = (X.shape[0] - FlightsClsDataset._default_train_num) / X.shape[0]
        idx_tr, idx_ts = equal_split(X.shape[0], train_frac)
        return X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class SusyDataset(RandomSplitDataset):
    file_name = '/data/DATASETS/SUSY/Susy.mat'
    dset_name = 'SUSY'
    default_train_frac = 0.8

    def read_data(self, dtype):
        with h5py.File(SusyDataset.file_name, "r") as f:
            arr = np.asarray(f['X'], dtype=as_np_dtype(dtype)).T
            X = arr[:, 1:]
            Y = arr[:, 0].reshape(-1, 1)
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class CIFAR10Dataset(KnownSplitDataset):
    file_name = "/data/DATASETS/CIFAR10/cifar10.mat"
    ts_file_name = "/data/DATASETS/CIFAR10/cifar10.t.mat"
    dset_name = "CIFAR10"
    num_train_samples = 50000

    def read_data(self, dtype):
        tr_data = scio.loadmat(CIFAR10Dataset.file_name)
        ts_data = scio.loadmat(CIFAR10Dataset.ts_file_name)
        X = np.concatenate((tr_data['Z'], ts_data['Z']), axis=0).astype(as_np_dtype(dtype))
        Y = np.concatenate((tr_data['y'], ts_data['y']), axis=0).astype(as_np_dtype(dtype))
        X = rgb_to_bw(X, dim=32)
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr / 255, Xts / 255, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_onehot(Ytr, Yts, num_classes=10)


class SVHNDataset(KnownSplitDataset):
    file_name = "/data/DATASETS/SVHN/SVHN.mat"
    ts_file_name = "/data/DATASETS/SVHN/SVHN.t.mat"
    dset_name = "SVHN"
    num_train_samples = 73257

    def read_data(self, dtype):
        tr_data = scio.loadmat(SVHNDataset.file_name)
        ts_data = scio.loadmat(SVHNDataset.ts_file_name)
        X = np.concatenate((tr_data['Z'], ts_data['Z']), axis=0).astype(as_np_dtype(dtype))
        Y = np.concatenate((tr_data['y'], ts_data['y']), axis=0).astype(as_np_dtype(dtype))
        X = rgb_to_bw(X, dim=32)
        Y = Y - 1  # Y is 1-indexed, convert to 0 index.
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr / 255, Xts / 255, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_onehot(Ytr, Yts, num_classes=10)


class FashionMnistDataset(KnownSplitDataset, Hdf5Dataset):
    file_name = "/data/DATASETS/misc/fashion_mnist.hdf5"
    dset_name = "FASHION_MNIST"
    num_train_samples = 60000

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        Xtr /= 255.0
        Xts /= 255.0
        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_onehot(Ytr, Yts, num_classes=10)


class MnistSmallDataset(KnownSplitDataset, Hdf5Dataset):
    file_name = "/data/DATASETS/misc/mnist.hdf5"
    dset_name = "MNIST"
    num_train_samples = 60000

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        Xtr /= 255.0
        Xts /= 255.0
        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_onehot(Ytr, Yts, num_classes=10)


class MnistDataset(KnownSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/MNIST/mnist8m_normalized.hdf5'
    dset_name = 'MNIST8M'
    num_train_samples = 6750000
    num_test_samples = 10_000

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_onehot(Ytr, Yts, num_classes=10, damping=True)


class SmallHiggsDataset(Hdf5Dataset, KnownSplitDataset):
    file_name = '/data/DATASETS/HIGGS_UCI/higgs_for_ho.hdf5'
    dset_name = 'HIGGSHO'
    num_train_samples = 10_000
    num_test_samples = 20_000

    def read_centers(self, dtype):
        with h5py.File(self.file_name, 'r') as h5py_file:
            centers = np.array(h5py_file['centers'], dtype=as_np_dtype(dtype))
        return centers

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        centers = self.read_centers(Xtr.dtype)

        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = np.var(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True).astype(Xtr.dtype)
        Xtr -= mtr
        Xtr /= vtr
        Xts -= mtr
        Xts /= vtr
        centers -= mtr
        centers /= vtr

        return Xtr, Xts, {'centers': centers}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class IctusDataset(RandomSplitDataset):
    file_name = '/data/DATASETS/ICTUS/run_all.mat'
    dset_name = 'ICTUS'
    default_train_frac = 0.8

    def read_data(self, dtype):
        data_dict = scio.loadmat(IctusDataset.file_name)
        X = np.asarray(data_dict['X'], dtype=as_np_dtype(dtype))
        Y = np.asarray(data_dict['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        mtr = np.mean(Xtr, axis=0, dtype=np.float64, keepdims=True).astype(Xtr.dtype)
        vtr = (1.0 / np.std(Xtr, axis=0, dtype=np.float64, ddof=1, keepdims=True)).astype(Xtr.dtype)

        Xtr -= mtr
        Xtr *= vtr
        Xts -= mtr
        Xts *= vtr

        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class SyntheticDataset(RandomSplitDataset):
    file_name = '/data/DATASETS/Synthetic0.1Noise.mat'
    dset_name = 'SYNTH01NOISE'
    default_train_frac = 0.5

    def read_data(self, dtype):
        data_dict = scio.loadmat(SyntheticDataset.file_name)
        X = np.asarray(data_dict['X'], dtype=as_np_dtype(dtype))
        Y = np.asarray(data_dict['Y'], dtype=as_np_dtype(dtype))
        return X, Y

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr.reshape((-1, 1)), Yts.reshape((-1, 1)), {}


class ChietDataset(KnownSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/weather/CHIET.hdf5'
    dset_name = 'CHIET'
    num_train_samples = 26227
    num_test_samples = 7832

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class EnergyDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/energy.hdf5'
    dset_name = 'ENERGY'
    default_train_frac = 0.8

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class BostonDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/boston.hdf5'
    dset_name = 'BOSTON'
    default_train_frac = 0.8

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class ProteinDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/protein.hdf5'
    dset_name = 'PROTEIN'
    default_train_frac = 0.8

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class Kin40kDataset(KnownSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/kin40k.hdf5'
    dset_name = 'KIN40K'
    num_train_samples = 10_000
    num_test_samples = 30_000

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class CodRnaDataset(KnownSplitDataset):
    folder = '/data/DATASETS/libsvm/binary'
    dset_name = 'cod-rna'
    num_train_samples = 59_535
    num_test_samples = 271_617

    def read_data(self, dtype):
        x_data, y_data = load_from_t(CodRnaDataset.dset_name, CodRnaDataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr.reshape(-1, 1), Yts.reshape(-1, 1), {}  # Is already -1, +1


class SvmGuide1Dataset(KnownSplitDataset):
    folder = '/data/DATASETS/libsvm/binary'
    dset_name = 'svmguide1'
    num_train_samples = 3089
    num_test_samples = 4000

    def read_data(self, dtype):
        x_data, y_data = load_from_t(SvmGuide1Dataset.dset_name, SvmGuide1Dataset.folder)
        x_data = x_data.astype(as_np_dtype(dtype))
        y_data = y_data.astype(as_np_dtype(dtype))
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class PhishingDataset(RandomSplitDataset):
    folder = '/data/DATASETS/libsvm/binary'
    dset_name = 'phishing'
    default_train_frac = 0.7

    def read_data(self, dtype):
        x_data, y_data = load_from_npz(self.dset_name, self.folder, dtype)
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}  # No preproc, all values are equal-.-

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 0, 1 -> -1, +1


class SpaceGaDataset(RandomSplitDataset):
    folder = '/data/DATASETS/libsvm/regression'
    dset_name = 'space_ga'
    default_train_frac = 0.7

    def read_data(self, dtype):
        x_data, y_data = load_from_npz(self.dset_name, self.folder, dtype)
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class CadataDataset(RandomSplitDataset):
    folder = '/data/DATASETS/libsvm/regression'
    dset_name = 'cadata'
    default_train_frac = 0.7

    def read_data(self, dtype):
        x_data, y_data = load_from_npz(self.dset_name, self.folder, dtype)
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class MgDataset(RandomSplitDataset):
    folder = '/data/DATASETS/libsvm/regression'
    dset_name = 'mg'
    default_train_frac = 0.7

    def read_data(self, dtype):
        x_data, y_data = load_from_npz(self.dset_name, self.folder, dtype)
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class CpuSmallDataset(RandomSplitDataset):
    folder = '/data/DATASETS/libsvm/regression'
    dset_name = 'cpusmall'
    default_train_frac = 0.7

    def read_data(self, dtype):
        x_data, y_data = load_from_npz(self.dset_name, self.folder, dtype)
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class AbaloneDataset(RandomSplitDataset):
    folder = '/data/DATASETS/libsvm/regression'
    dset_name = 'abalone'
    default_train_frac = 0.7

    def read_data(self, dtype):
        x_data, y_data = load_from_npz(self.dset_name, self.folder, dtype)
        return x_data, y_data

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class CaspDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/casp.hdf5'
    dset_name = 'casp'
    default_train_frac = 0.7

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class BlogFeedbackDataset(KnownSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/BlogFeedback.hdf5'
    dset_name = 'blog-feedback'
    num_train_samples = 52397

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class CovTypeDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/covtype_binary.hdf5'
    dset_name = 'covtype'
    default_train_frac = 0.7

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return convert_to_binary_y(Ytr, Yts)  # 1, 2 -> -1, +1


class Ijcnn1Dataset(KnownSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/ijcnn1.hdf5'
    dset_name = 'ijcnn1'
    num_train_samples = 49990

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Xtr, Xts, {}  # Data already standardized

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return Ytr.reshape(-1, 1), Yts.reshape(-1, 1), {}  # binary-classif : already -1, +1


class BuzzDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/buzz.hdf5'
    dset_name = 'buzz'
    default_train_frac = 0.7
    dset_shape = (583250, 77)

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Weird preprocessing from AGW
        Ytr = np.log(Ytr + 1.0)
        Yts = np.log(Yts + 1.0)
        return standardize_y(Ytr, Yts)


class Road3DDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/3droad.hdf5'
    dset_name = '3DRoad'
    default_train_frac = 0.7
    dset_shape = (434874, 3)

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_y(Ytr, Yts)


class HouseEelectricDataset(RandomSplitDataset, Hdf5Dataset):
    file_name = '/data/DATASETS/misc/houseelectric.hdf5'
    dset_name = 'HouseElectric'
    default_train_frac = 0.7
    dset_shape = (2049280, 11)

    def preprocess_x(self, Xtr: np.ndarray, Xts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        return standardize_x(Xtr, Xts)

    def preprocess_y(self, Ytr: np.ndarray, Yts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Weird preprocessing from AGW
        Ytr = np.log(Ytr)
        Yts = np.log(Yts)
        return standardize_y(Ytr, Yts)


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
    Dataset.HOHIGGS: SmallHiggsDataset(),
    Dataset.ICTUS: IctusDataset(),
    Dataset.SYNTH01NOISE: SyntheticDataset(),
    Dataset.CHIET: ChietDataset(),
    Dataset.ENERGY: EnergyDataset(),
    Dataset.BOSTON: BostonDataset(),
    Dataset.PROTEIN: ProteinDataset(),
    Dataset.KIN40K: Kin40kDataset(),
    Dataset.CODRNA: CodRnaDataset(),
    Dataset.SVMGUIDE1: SvmGuide1Dataset(),
    Dataset.PHISHING: PhishingDataset(),
    Dataset.SPACEGA: SpaceGaDataset(),
    Dataset.CADATA: CadataDataset(),
    Dataset.MG: MgDataset(),
    Dataset.CPUSMALL: CpuSmallDataset(),
    Dataset.ABALONE: AbaloneDataset(),
    Dataset.CASP: CaspDataset(),
    Dataset.BLOGFEEDBACK: BlogFeedbackDataset(),
    Dataset.COVTYPE: CovTypeDataset(),
    Dataset.IJCNN1: Ijcnn1Dataset(),
    Dataset.FASHION_MNIST: FashionMnistDataset(),
    Dataset.BUZZ: BuzzDataset(),
    Dataset.ROAD3D: Road3DDataset(),
    Dataset.HOUSEELECTRIC: HouseEelectricDataset(),
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
