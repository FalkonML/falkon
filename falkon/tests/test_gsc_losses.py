import pytest
import numpy as np
import torch

from falkon import kernels
from falkon.gsc_losses import (
    LogisticLoss,
    WeightedCrossEntropyLoss,
)


def naive_logistic_loss(true, pred):
    return torch.log(1 + torch.exp(-true * pred))


def naive_bce(true, pred, weight):
    return -(true * torch.log(torch.sigmoid(pred)) + weight * (1 - true) * torch.log(1 - torch.sigmoid(pred)))


def derivative_test(diff_fn, loss, pred, true):
    o_true, o_pred = true.clone(), pred.clone()
    exp = diff_fn(true, pred)

    exp_d = [
        torch.autograd.grad(exp[i], pred, retain_graph=True, create_graph=True)[0][i]
        for i in range(pred.shape[0])
    ]
    exp_dd = [
        torch.autograd.grad(exp_d[i], pred, retain_graph=True)[0][i]
        for i in range(pred.shape[0])
    ]

    pred = pred.detach()
    np.testing.assert_allclose(exp.detach().numpy(), loss(true, pred).detach().numpy())
    np.testing.assert_allclose(o_true.detach(), true.detach())
    np.testing.assert_allclose(o_pred.detach(), pred.detach())
    np.testing.assert_allclose([e.item() for e in exp_d], loss.df(true, pred).detach().numpy())
    np.testing.assert_allclose(o_true.detach(), true.detach())
    np.testing.assert_allclose(o_pred.detach(), pred.detach())
    np.testing.assert_allclose([e.item() for e in exp_dd], loss.ddf(true, pred).detach().numpy())
    np.testing.assert_allclose(o_true.detach(), true.detach())
    np.testing.assert_allclose(o_pred.detach(), pred.detach())


def test_logistic_loss_derivative():
    kernel = kernels.GaussianKernel(3)
    pred = torch.linspace(-10, 10, 10, dtype=torch.float64).requires_grad_()
    true = torch.bernoulli(torch.ones_like(pred, dtype=torch.float64) / 2) * 2 - 1  # -1, +1 random values
    log_loss = LogisticLoss(kernel)

    derivative_test(naive_logistic_loss, log_loss, pred, true)


def test_bce_derivative():
    kernel = kernels.GaussianKernel(3)
    pred = torch.linspace(-10, 10, 10, dtype=torch.float64).requires_grad_()
    true = torch.bernoulli(torch.ones_like(pred, dtype=torch.float64) / 2)  # 0, +1 random values

    neg_weight = 1
    wbce_loss = WeightedCrossEntropyLoss(kernel, neg_weight=neg_weight)

    derivative_test(lambda t, p: naive_bce(t, p, neg_weight), wbce_loss, pred, true)


def test_weighted_bce_derivative():
    kernel = kernels.GaussianKernel(3)
    pred = torch.linspace(-10, 10, 10, dtype=torch.float64).requires_grad_()
    true = torch.bernoulli(torch.ones_like(pred, dtype=torch.float64) / 2)  # 0, +1 random values

    neg_weight = 0.5
    wbce_loss = WeightedCrossEntropyLoss(kernel, neg_weight=neg_weight)

    derivative_test(lambda t, p: naive_bce(t, p, neg_weight), wbce_loss, pred, true)


if __name__ == "__main__":
    pytest.main()
