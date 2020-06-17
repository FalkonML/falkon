import unittest

import numpy as np
import torch

from falkon.precond import FalkonPreconditioner
from falkon.kernels import GaussianKernel
from falkon.center_selection import UniformSel
from falkon.optim.conjgrad import ConjugateGradient, FalkonConjugateGradient
from falkon.tests.helpers import gen_random, gen_random_pd


class TestConjugateGradient(unittest.TestCase):
    def test_one_rhs(self):
        t = 200
        A = torch.from_numpy(gen_random_pd(t, 'float64', F=False, seed=9))
        b = torch.from_numpy(gen_random(t, 1, 'float64', F=False, seed=10))
        # Solve Ax = b for x
        opt = ConjugateGradient()
        x = opt.solve(X0=None, B=b, mmv=lambda x: A@x, max_iter=10, callback=None)

        self.assertEqual((200, 1), x.shape)
        expected = np.linalg.solve(A.numpy(), b.numpy())
        np.testing.assert_allclose(expected, x)

    def test_multi_rhs(self):
        t = 200
        A = torch.from_numpy(gen_random_pd(t, 'float64', F=False, seed=10))
        b = torch.from_numpy(gen_random(t, 10, 'float64', F=False, seed=14))
        # Solve Ax = b for x
        opt = ConjugateGradient()
        x = opt.solve(X0=None, B=b, mmv=lambda x: A@x, max_iter=200, callback=None)

        self.assertEqual((t, 10), x.shape)
        expected = np.linalg.solve(A.numpy(), b.numpy())
        np.testing.assert_allclose(expected, x)

    def test_fortran(self):
        t = 200
        opt = {
            'no_keops': True,  # Cannot use keops with Fortran-contiguous
        }
        A = torch.from_numpy(gen_random_pd(t, 'float64', F=True, seed=10))
        b = torch.from_numpy(gen_random(t, 10, 'float64', F=True, seed=14))
        # Solve Ax = b for x
        opt = ConjugateGradient(opt=opt)
        x = opt.solve(X0=None, B=b, mmv=lambda x: A@x, max_iter=200, callback=None)

        self.assertEqual((t, 10), x.shape)
        expected = np.linalg.solve(A.numpy(), b.numpy())
        np.testing.assert_allclose(expected, x)


class TestFalkonConjugateGradient(unittest.TestCase):
    def setUp(self):
        self.M = 10
        self.N = 500
        self.kernel = GaussianKernel(100.0)
        self.opt = {'compute_arch_speed': False,
                    'final_type': torch.float64,
                    'inter_type': torch.float64,
                    'use_cpu': True}
        self.X = torch.from_numpy(gen_random(self.N, 10, 'float64', F=True, seed=10))
        cs = UniformSel(np.random.default_rng(2))
        self.centers = cs.select(self.X, self.M, )
        self.knm = self.kernel(self.X, self.centers, opt=self.opt)
        self.kmm = self.kernel(self.centers, self.centers, opt=self.opt)

        self.la = 10
        self.prec = FalkonPreconditioner(self.la, self.kernel, self.opt)
        self.prec.init(self.centers)

    def test_one_rhs(self):
        b = torch.from_numpy(gen_random(self.N, 1, 'float64', F=True, seed=10))
        opt = FalkonConjugateGradient(self.kernel, self.prec, opt=self.opt)

        # Solve (knm.T @ knm + lambda*n*kmm) x = knm.T @ b
        rhs = self.knm.T @ b
        lhs = self.knm.T @ self.knm + self.la * self.N * self.kmm
        expected = np.linalg.solve(lhs.numpy(), rhs.numpy())

        beta = opt.solve(self.X, self.centers, b, self.la, None, 20)
        alpha = self.prec.apply(beta)

        np.testing.assert_allclose(expected, alpha, rtol=1e-6)

