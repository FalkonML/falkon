import numpy as np
import pytest
import torch
from falkon.models.incore_falkon import InCoreFalkon
from sklearn import datasets

from falkon import Falkon, kernels
from falkon.options import FalkonOptions
from falkon.utils import decide_cuda


@pytest.fixture
def cls_data():
    X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64).reshape(-1, 1)
    Y[Y == 0] = -1
    return torch.from_numpy(X), torch.from_numpy(Y)


@pytest.fixture
def multicls_data():
    X, Y = datasets.make_classification(2000, 30, n_classes=20, n_informative=6, random_state=11)
    X = X.astype(np.float64)
    eye = np.eye(20, dtype=np.float64)
    Y = eye[Y.astype(np.int32), :]
    return torch.from_numpy(X), torch.from_numpy(Y)


@pytest.fixture
def reg_data():
    X, Y = datasets.make_regression(1000, 30, random_state=11)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64).reshape(-1, 1)
    return torch.from_numpy(X[:800]), torch.from_numpy(Y[:800]), \
        torch.from_numpy(X[800:]), torch.from_numpy(Y[800:])


class TestFalkon:
    def test_classif(self, cls_data):
        X, Y = cls_data
        kernel = kernels.GaussianKernel(2.0)

        def error_fn(t, p):
            return 100 * torch.sum(t * p <= 0).to(torch.float32) / t.shape[0], "c-err"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)

        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        flk.fit(X, Y)
        preds = flk.predict(X)
        err = error_fn(preds, Y)[0]
        assert err < 5

    def test_multiclass(self, multicls_data):
        X, Y = multicls_data
        kernel = kernels.GaussianKernel(10.0)

        def error_fn(t, p):
            t = torch.argmax(t, dim=1)
            p = torch.argmax(p, dim=1)
            return torch.mean((t.reshape(-1, ) != p.reshape(-1, )).to(torch.float64)), "multic-err"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)

        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        flk.fit(X, Y)
        preds = flk.predict(X)
        err = error_fn(preds, Y)[0]
        assert err < 0.23

    def test_regression(self, reg_data):
        Xtr, Ytr, Xts, Yts = reg_data
        kernel = kernels.GaussianKernel(20.0)

        def error_fn(t, p):
            return torch.sqrt(torch.mean((t - p) ** 2)), "RMSE"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)

        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        flk.fit(Xtr, Ytr, Xts=Xts, Yts=Yts)

        assert flk.predict(Xts).shape == (Yts.shape[0], 1)
        ts_err = error_fn(flk.predict(Xts), Yts)[0]
        tr_err = error_fn(flk.predict(Xtr), Ytr)[0]
        assert tr_err < ts_err
        assert ts_err < 2.5

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cuda_predict(self, reg_data):
        Xtr, Ytr, Xts, Yts = reg_data
        kernel = kernels.GaussianKernel(20.0)

        def error_fn(t, p):
            return torch.sqrt(torch.mean((t - p) ** 2)), "RMSE"

        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=True,
                min_cuda_pc_size_64=1, min_cuda_iter_size_64=1)

        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        flk.fit(Xtr, Ytr, Xts=Xts, Yts=Yts)
        flk.to("cuda:0")

        cuda_preds = flk.predict(Xts.to("cuda:0"))
        assert cuda_preds.device.type == "cuda"
        assert cuda_preds.shape == (Yts.shape[0], 1)
        ts_err = error_fn(cuda_preds.cpu(), Yts)[0]
        tr_err = error_fn(cuda_preds.cpu(), Ytr)[0]
        assert tr_err < ts_err
        assert ts_err < 2.5


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestIncoreFalkon:
    def test_fails_cpu_tensors(self, cls_data):
        X, Y = cls_data
        kernel = kernels.GaussianKernel(2.0)

        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=True)

        flk = InCoreFalkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt)
        with pytest.raises(ValueError):
            flk.fit(X, Y)
        flk.fit(X.cuda(), Y.cuda())
        with pytest.raises(ValueError):
            flk.predict(X)

    def test_classif(self, cls_data):
        X, Y = cls_data
        X = X.cuda()
        Y = Y.cuda()
        kernel = kernels.GaussianKernel(2.0)

        def error_fn(t, p):
            return 100 * torch.sum(t * p <= 0).to(torch.float32) / t.shape[0], "c-err"

        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=True)

        flk = InCoreFalkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        flk.fit(X, Y)
        preds = flk.predict(X)
        assert preds.device == X.device
        err = error_fn(preds, Y)[0]
        assert err < 5