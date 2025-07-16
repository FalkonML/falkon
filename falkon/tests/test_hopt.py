import pytest
import torch

import falkon.kernels
from falkon import FalkonOptions
from falkon.hopt.objectives import GCV, LOOCV, SGPR, CompReg, HoldOut, NystromCompReg, StochasticNystromCompReg
from falkon.kernels import GaussianKernel, PolynomialKernel

n, d = 1000, 10


@pytest.fixture(params=["gauss"], scope="function")
def kernel(request) -> falkon.kernels.Kernel:
    if request.param == "gauss":
        return GaussianKernel(sigma=torch.tensor([5.0] * d, dtype=torch.float32, requires_grad=True))
    if request.param == "poly":
        return PolynomialKernel(
            beta=torch.tensor(3.0, requires_grad=True),
            gamma=torch.tensor(1.0, requires_grad=False),
            degree=torch.tensor(2, requires_grad=False),
        )
    raise ValueError("Unmatched request parameter.")


def init_model(model_cls, kernel, centers_init, penalty_init, opt_centers, opt_penalty):
    if model_cls == HoldOut:
        return HoldOut(kernel, centers_init, penalty_init, opt_centers, opt_penalty, val_pct=0.8, per_iter_split=False)
    return model_cls(kernel, centers_init, penalty_init, opt_centers, opt_penalty)


@pytest.mark.parametrize("model_cls", [CompReg, NystromCompReg, SGPR, GCV, LOOCV, HoldOut])
def test_exact_objectives(model_cls, kernel):
    # Generate some synthetic data
    dtype = torch.float64
    torch.manual_seed(12)
    num_centers = 100
    X = torch.randn((n, d), dtype=dtype)
    w = torch.arange(X.shape[1], dtype=dtype).reshape(-1, 1)
    Y = X @ w + torch.randn((X.shape[0], 1), dtype=dtype) * 0.3  # 10% noise
    num_train = int(X.shape[0] * 0.8)
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]
    # Standardize X, Y
    x_mean, x_std = torch.mean(X_train, dim=0, keepdim=True), torch.std(X_train, dim=0, keepdim=True)
    y_mean, y_std = torch.mean(Y_train, dim=0, keepdim=True), torch.std(Y_train, dim=0, keepdim=True)
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    Y_train = (Y_train - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std

    # sigma_init = torch.tensor([5.0] * X_train.shape[1], dtype=torch.float32)
    penalty_init = torch.tensor(1e-5, dtype=dtype)
    centers_init = X_train[:num_centers].clone()

    model = init_model(model_cls, kernel, centers_init, penalty_init, True, True)
    # kernel = GaussianKernel(sigma_init.requires_grad_())
    # model = CompReg(kernel, centers_init, penalty_init, True, True)
    opt_hp = torch.optim.Adam(model.parameters(), lr=0.1)

    from torch import autograd

    autograd.set_detect_anomaly(True)
    for _ in range(50):
        # print(model.kernel.beta, model.kernel.gamma, model.kernel.degree)
        opt_hp.zero_grad()
        loss = model(X_train, Y_train)
        print("Loss", loss)
        loss.backward()
        opt_hp.step()
    ts_err = torch.mean((model.predict(X_test) - Y_test) ** 2)
    print(f"Model {model_cls} obtains {ts_err:.4f} error")
    # assert ts_err < 300


def test_stoch_objectives(kernel):
    # Generate some synthetic data
    torch.manual_seed(12)
    num_centers = 100
    dtype = torch.float32
    X = torch.randn((n, d), dtype=dtype)
    w = torch.arange(X.shape[1], dtype=dtype).reshape(-1, 1)
    Y = X @ w + torch.randn((X.shape[0], 1), dtype=dtype) * 0.1  # 10% noise

    num_train = int(X.shape[0] * 0.8)
    X_train, Y_train = X[:num_train], Y[:num_train]
    X_test, Y_test = X[num_train:], Y[num_train:]
    # Standardize X, Y
    x_mean, x_std = torch.mean(X_train, dim=0, keepdim=True), torch.std(X_train, dim=0, keepdim=True)
    y_mean, y_std = torch.mean(Y_train, dim=0, keepdim=True), torch.std(Y_train, dim=0, keepdim=True)
    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    Y_train = (Y_train - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std

    penalty_init = torch.tensor(1e-1, dtype=dtype)
    centers_init = X_train[:num_centers].clone()
    flk_opt = FalkonOptions(
        use_cpu=True, keops_active="no", pc_epsilon_32=1e-4, cg_tolerance=1e-4, cg_differential_convergence=True
    )

    model = StochasticNystromCompReg(
        kernel, centers_init, penalty_init, True, True, flk_opt=flk_opt, flk_maxiter=100, num_trace_est=10
    )
    opt_hp = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(100):
        opt_hp.zero_grad()
        loss = model(X_train, Y_train)
        loss.backward()
        print("Loss", loss.item())
        opt_hp.step()
    ts_err = torch.mean((model.predict(X_test) - Y_test) ** 2)
    print(f"Model {model.__class__} obtains {ts_err:.4f} error")
    assert ts_err < 60
    # assert ts_err < 300
