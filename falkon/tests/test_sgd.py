import unittest

import numpy as np
import torch

from falkon.optim.sgd import SGD
from falkon.tests.helpers import gen_random


class TestSGD(unittest.TestCase):
    def test_one_rhs(self):
        t = 3000
        d = 8
        A = torch.from_numpy(gen_random(t, d, 'float64', F=False, seed=9))
        b = torch.from_numpy(gen_random(t, 1, 'float64', F=False, seed=10))

        # Solve Ax = b for x, with squared error gradient
        expected = np.linalg.solve((A.T@A).numpy(), (A.T@b).numpy())

        opt = SGD(mb_size=100, step_size=0.00001)
        X0 = torch.zeros(d, 1, dtype=torch.float64)
        def solve_mb(u, idx):
            out = A[idx,:].T @ A[idx,:] @ u - A[idx,:].T @ b[idx,:]
            return out
        approx_err = []
        def cback(i, x, t, t2):
            approx_err.append(np.mean(np.abs(x.numpy() - expected)))
        x = opt.solve(X0=X0, mmv=solve_mb, num_points=t, max_iter=20, callback=cback)

        dec_error = all(x>=y for x, y in zip(approx_err, approx_err[1:]))
        self.assertTrue(dec_error)

        sgdb = A@x
        expb = A@torch.from_numpy(expected)

        print("SGD ERR", torch.mean((sgdb - b)**2))
        print("REAL ERR", torch.mean((expb - b)**2))

    def test_multi_rhs(self):
        t = 3000
        d = 8
        A = torch.from_numpy(gen_random(t, d, 'float64', F=False, seed=9))
        b = torch.from_numpy(gen_random(t, 10, 'float64', F=False, seed=10))

        # Solve Ax = b for x, with squared error gradient
        expected = np.linalg.solve((A.T@A).numpy(), (A.T@b).numpy())

        opt = SGD(mb_size=200, step_size=0.00001)
        X0 = torch.zeros(d, 10, dtype=torch.float64)
        def solve_mb(u, idx):
            out = A[idx,:].T @ A[idx,:] @ u - A[idx,:].T @ b[idx,:]
            return out
        approx_err = []
        def cback(i, x, t, t2):
            approx_err.append(np.mean(np.abs(x.numpy() - expected)))
        x = opt.solve(X0=X0, mmv=solve_mb, num_points=t, max_iter=100, callback=cback)

        dec_error = all(x>=y for x, y in zip(approx_err, approx_err[1:]))
        self.assertTrue(dec_error)

        sgdb = A@x
        expb = A@torch.from_numpy(expected)

        print("SGD ERR", torch.mean((sgdb - b)**2))
        print("REAL ERR", torch.mean((expb - b)**2))



if __name__ == "__main__":
    unittest.main()
