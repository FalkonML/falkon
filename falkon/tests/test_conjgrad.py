import numpy as np
import pytest
import torch

from falkon.center_selection import UniformSel
from falkon.kernels import GaussianKernel
from falkon.optim.conjgrad import ConjugateGradient, FalkonConjugateGradient
from falkon.options import FalkonOptions
from falkon.preconditioner import FalkonPreconditioner
from falkon.tests.gen_random import gen_random, gen_random_pd


class TestConjugateGradient():
    t = 200

    @pytest.fixture()
    def mat(self):
        return torch.from_numpy(gen_random_pd(self.t, 'float64', F=False, seed=9))

    @pytest.fixture()
    def conjgrad(self):
        return ConjugateGradient()

    @pytest.fixture(params=[1, 10], ids=["1-rhs", "10-rhs"])
    def vec_rhs(self, request):
        return torch.from_numpy(gen_random(self.t, request.param, 'float64', F=False, seed=9))

    @pytest.mark.parametrize("order", ["F", "C"])
    def test_one_rhs(self, mat, vec_rhs, conjgrad, order):
        if order == "F":
            mat = torch.from_numpy(np.asfortranarray(mat.numpy()))
            vec_rhs = torch.from_numpy(np.asfortranarray(vec_rhs.numpy()))

        x = conjgrad.solve(X0=None, B=vec_rhs, mmv=lambda x: mat@x, max_iter=10, callback=None)

        assert x.shape == (self.t, vec_rhs.shape[1])
        expected = np.linalg.solve(mat.numpy(), vec_rhs.numpy())
        np.testing.assert_allclose(expected, x, rtol=1e-6)


# TODO: KeOps fails if data is F-contig. Check if this occurs also in `test_falkon`.
#       if so we need to fix in the falkon fn.
class TestFalkonConjugateGradient:
    basic_opt = FalkonOptions(use_cpu=True)
    N = 500
    M = 10
    D = 10
    penalty = 10

    @pytest.fixture()
    def kernel(self):
        return GaussianKernel(100.0)

    @pytest.fixture()
    def data(self):
        return torch.from_numpy(gen_random(self.N, self.D, 'float64', F=False, seed=10))

    @pytest.fixture(params=[1, 10], ids=["1-rhs", "10-rhs"])
    def vec_rhs(self, request):
        return torch.from_numpy(gen_random(self.N, request.param, 'float64', F=False, seed=9))

    @pytest.fixture()
    def centers(self, data):
        cs = UniformSel(np.random.default_rng(2))
        return cs.select(data, None, self.M)

    @pytest.fixture()
    def knm(self, kernel, data, centers):
        return kernel(data, centers, opt=self.basic_opt)

    @pytest.fixture()
    def kmm(self, kernel, centers):
        return kernel(centers, centers, opt=self.basic_opt)

    @pytest.fixture()
    def preconditioner(self, kernel, centers):
        prec = FalkonPreconditioner(self.penalty, kernel, self.basic_opt)
        prec.init(centers)
        return prec

    def test_flk_cg(self, data, centers, kernel, preconditioner, knm, kmm, vec_rhs):
        opt = FalkonConjugateGradient(kernel, preconditioner, opt=self.basic_opt)

        # Solve (knm.T @ knm + lambda*n*kmm) x = knm.T @ b
        rhs = knm.T @ vec_rhs
        lhs = knm.T @ knm + self.penalty * self.N * kmm
        expected = np.linalg.solve(lhs.numpy(), rhs.numpy())

        beta = opt.solve(data, centers, vec_rhs, self.penalty, None, 200)
        alpha = preconditioner.apply(beta)

        np.testing.assert_allclose(expected, alpha, rtol=1e-5)
