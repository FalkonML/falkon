import unittest

import numpy as np
import torch

from falkon import kernels

from falkon.gsc_losses import LogisticLoss
from falkon.logistic_falkon import LogisticFalkon
from falkon.tests.helpers import gen_random


class TestLogisticFalkon(unittest.TestCase):
    def test_simple(self):
        from sklearn import datasets
        X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        Y[Y == 0] = -1

        kernel = kernels.GaussianKernel(3.0)
        loss = LogisticLoss(kernel=kernel)
        def error_fn(t, p):
            return 100 * torch.sum(t*p <= 0) / t.shape[0], "c-err"

        logflk = LogisticFalkon(
            kernel=kernel, loss=loss, penalty_list=[1e-1, 1e-3, 1e-5, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
            iter_list=[3, 3, 3, 3, 8, 8, 8, 8], M=500, seed=10, use_cpu=True, no_keops=True, debug=False,
            error_fn=error_fn)
        logflk.fit(torch.from_numpy(X), torch.from_numpy(Y))

class TestOriginal(unittest.TestCase):
    def test_simple(self):
        import sys
        sys.path.append("/home/giacomo/Dropbox/unige/falkon/Newton-Method-for-GSC-losses-/SECOND ORDER METHOD")
        from NewtonMethod import NewtonMethod
        import gaussianKernel
        import losses
        import display as dp
        kern = gaussianKernel.gaussianKernel(1.0)

        from sklearn import datasets
        X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
        X = torch.from_numpy(X.astype(np.float64))
        Y = torch.from_numpy(Y.astype(np.float64)).reshape(-1, 1)
        Y[Y == 0] = -1

        loss = losses.logloss

        #Parameters for Newton Method
        la_list = [1e-1, 1e-4, 1e-7, 1e-7]
        t_list = [3, 3, 3, 8]

        #Perform Newton method
        alpha = NewtonMethod(loss,X,X[:100],Y,Y[:100],kern,la_list,t_list, useGPU=False, cobj=dp.cobjTest(X, Y))



if __name__ == '__main__':
    unittest.main()
