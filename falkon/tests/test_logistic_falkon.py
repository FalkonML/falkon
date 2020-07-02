import numpy as np
import pytest
import torch
from sklearn import datasets

from falkon import kernels
from falkon.gsc_losses import LogisticLoss
from falkon.models.logistic_falkon import LogisticFalkon
from falkon.options import FalkonOptions


@pytest.fixture
def data():
    X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64).reshape(-1, 1)
    Y[Y == 0] = -1
    return torch.from_numpy(X), torch.from_numpy(Y)


class TestLogisticFalkon:
    def test_simple(self, data):
        X, Y = data
        kernel = kernels.GaussianKernel(3.0)
        loss = LogisticLoss(kernel=kernel)

        def error_fn(t, p):
            return 100 * torch.sum(t*p <= 0) / t.shape[0], "c-err"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)

        logflk = LogisticFalkon(
            kernel=kernel, loss=loss, penalty_list=[1e-1, 1e-3, 1e-5, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
            iter_list=[3, 3, 3, 3, 8, 8, 8, 8], M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        logflk.fit(X, Y)
        preds = logflk.predict(X)
        err = error_fn(preds, Y)[0]
        assert err < 0.1