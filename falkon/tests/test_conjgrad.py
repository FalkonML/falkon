import dataclasses

import numpy as np
import pytest
import torch

from falkon.utils.tensor_helpers import move_tensor, create_same_stride
from falkon.utils import decide_cuda
from falkon.center_selection import UniformSelector
from falkon.kernels import GaussianKernel
from falkon.optim.conjgrad import ConjugateGradient, FalkonConjugateGradient
from falkon.options import FalkonOptions
from falkon.preconditioner import FalkonPreconditioner
from falkon.tests.gen_random import gen_random, gen_random_pd


@pytest.mark.full
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))])
class TestConjugateGradient:
    t = 200

    @pytest.fixture()
    def mat(self):
        return gen_random_pd(self.t, 'float64', F=False, seed=9)

    @pytest.fixture()
    def conjgrad(self):
        return ConjugateGradient()

    @pytest.fixture(params=[1, 10], ids=["1-rhs", "10-rhs"])
    def vec_rhs(self, request):
        return torch.from_numpy(gen_random(self.t, request.param, 'float64', F=False, seed=9))

    def test_one_rhs(self, mat, vec_rhs, conjgrad, order, device):
        if order == "F":
            mat = torch.from_numpy(np.asfortranarray(mat.numpy()))
            vec_rhs = torch.from_numpy(np.asfortranarray(vec_rhs.numpy()))
        mat = move_tensor(mat, device)
        vec_rhs = move_tensor(vec_rhs, device)

        x = conjgrad.solve(X0=None, B=vec_rhs, mmv=lambda x_: mat @ x_, max_iter=10, callback=None)

        assert str(x.device) == device, "Device has changed unexpectedly"
        assert x.stride() == vec_rhs.stride(), "Stride has changed unexpectedly"
        assert x.shape == (self.t, vec_rhs.shape[1]), "Output shape is incorrect"
        expected = np.linalg.solve(mat.cpu().numpy(), vec_rhs.cpu().numpy())
        np.testing.assert_allclose(expected, x.cpu().numpy(), rtol=1e-6)

    def test_with_x0(self, mat, vec_rhs, conjgrad, order, device):
        if order == "F":
            mat = torch.from_numpy(np.asfortranarray(mat.numpy()))
            vec_rhs = torch.from_numpy(np.asfortranarray(vec_rhs.numpy()))
        mat = move_tensor(mat, device)
        vec_rhs = move_tensor(vec_rhs, device)
        init_sol = create_same_stride(vec_rhs.size(), vec_rhs, vec_rhs.dtype, device)
        init_sol.fill_(0.0)

        x = conjgrad.solve(X0=init_sol, B=vec_rhs, mmv=lambda x_: mat @ x_, max_iter=10,
                           callback=None)

        assert x.data_ptr() == init_sol.data_ptr(), "Initial solution vector was copied"
        assert str(x.device) == device, "Device has changed unexpectedly"
        assert x.shape == (self.t, vec_rhs.shape[1]), "Output shape is incorrect"
        assert x.stride() == vec_rhs.stride(), "Stride has changed unexpectedly"
        expected = np.linalg.solve(mat.cpu().numpy(), vec_rhs.cpu().numpy())
        np.testing.assert_allclose(expected, x.cpu().numpy(), rtol=1e-6)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not decide_cuda(), reason="No GPU found."))])
class TestFalkonConjugateGradient:
    basic_opt = FalkonOptions(use_cpu=True, keops_active="no")
    N = 500
    M = 10
    D = 1000
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
        cs = UniformSelector(np.random.default_rng(2), num_centers=self.M)
        return cs.select(data, None)

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

    def test_flk_cg(self, data, centers, kernel, preconditioner, knm, kmm, vec_rhs, device):
        preconditioner = preconditioner.to(device)
        options = dataclasses.replace(self.basic_opt, use_cpu=device == "cpu")
        opt = FalkonConjugateGradient(kernel, preconditioner, opt=options)

        # Solve (knm.T @ knm + lambda*n*kmm) x = knm.T @ b
        rhs = knm.T @ vec_rhs
        lhs = knm.T @ knm + self.penalty * self.N * kmm
        expected = np.linalg.solve(lhs.numpy(), rhs.numpy())

        data = move_tensor(data, device)
        centers = move_tensor(centers, device)
        vec_rhs = move_tensor(vec_rhs, device)

        beta = opt.solve(X=data, M=centers, Y=vec_rhs, _lambda=self.penalty,
                         initial_solution=None, max_iter=100)
        alpha = preconditioner.apply(beta)

        assert str(beta.device) == device, "Device has changed unexpectedly"
        np.testing.assert_allclose(expected, alpha.cpu().numpy(), rtol=1e-5)

    def test_restarts(self, data, centers, kernel, preconditioner, knm, kmm, vec_rhs, device):
        preconditioner = preconditioner.to(device)
        options = dataclasses.replace(self.basic_opt, use_cpu=device == "cpu", cg_tolerance=1e-10)
        opt = FalkonConjugateGradient(kernel, preconditioner, opt=options)

        # Solve (knm.T @ knm + lambda*n*kmm) x = knm.T @ b
        rhs = knm.T @ vec_rhs
        lhs = knm.T @ knm + self.penalty * self.N * kmm
        expected = np.linalg.solve(lhs.numpy(), rhs.numpy())

        data = move_tensor(data, device)
        centers = move_tensor(centers, device)
        vec_rhs = move_tensor(vec_rhs, device)

        sol = None
        for _ in range(30):
            sol = opt.solve(X=data, M=centers, Y=vec_rhs, _lambda=self.penalty,
                            initial_solution=sol, max_iter=6)
            print()

        alpha = preconditioner.apply(sol)
        np.testing.assert_allclose(expected, alpha.cpu().numpy(), rtol=1e-5)

    def test_precomputed_kernel(self, data, centers, kernel, preconditioner, knm, kmm, vec_rhs, device):
        preconditioner = preconditioner.to(device)
        options = dataclasses.replace(self.basic_opt, use_cpu=device == "cpu")
        opt = FalkonConjugateGradient(kernel, preconditioner, opt=options)

        # Solve (knm.T @ knm + lambda*n*kmm) x = knm.T @ b
        rhs = knm.T @ vec_rhs
        lhs = knm.T @ knm + self.penalty * self.N * kmm
        expected = np.linalg.solve(lhs.numpy(), rhs.numpy())

        knm = move_tensor(knm, device)
        vec_rhs = move_tensor(vec_rhs, device)

        beta = opt.solve(X=knm, M=None, Y=vec_rhs, _lambda=self.penalty,
                         initial_solution=None, max_iter=200)
        alpha = preconditioner.apply(beta)

        assert str(beta.device) == device, "Device has changed unexpectedly"
        np.testing.assert_allclose(expected, alpha.cpu().numpy(), rtol=1e-5)
