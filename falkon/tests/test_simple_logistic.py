import unittest

from scipy.spatial.distance import cdist
import torch
import numpy as np

from falkon.gsc_losses import LogisticLoss
from falkon.kernels import GaussianKernel
import sys

from falkon.optim import ConjugateGradient
from falkon.precond import LogisticPreconditioner

sys.path.append("/home/giacomo/unige/falkon/FALKON")


# def naive_gaussian_kernel(X1, X2, sigma):
#     pairwise_dists = cdist(X1, X2, 'sqeuclidean')
#     return np.exp(-pairwise_dists / (2 * sigma ** 2))

#
def f(y, x):
    return torch.log(1 + torch.exp(-y * x))


def sigma(y, x):
    return 1 / (1 + torch.exp(-y * x))


def df(y, x):
    return -y * sigma(-y, x)


def ddf(y, x):
    return y ** 2 * sigma(-y, x) * sigma(y, x)


def tr_solve(x, T, transpose=False):
    return torch.triangular_solve(x, T, upper=True, transpose=transpose)[0]


def aux_fun(l: LogisticLoss, lobj, u, Kr, a, b):
    p = Kr @ u
    lobj[1] = p[a:b]
    return l(lobj[0][a:b], p).sum() / Kr.shape[0]


def aux_grad(l: LogisticLoss, lobj, u, Kr, a, b):
    p = Kr @ u
    lobj[1][a:b] = p
    _df = l.df(lobj[0][a:b], p)
    return (_df.t() @ Kr).t() / Kr.shape[0]


def aux_hess(l: LogisticLoss, lobj, u, Kr, a, b):
    p = Kr @ u
    return ((l.ddf(lobj[0][a:b], lobj[1][a:b]) * p).t() @ Kr).t() / Kr.shape[0]


def aux_mmv(l: LogisticLoss, lobj, u, Kr, a, b):
    lobj[1][a:b] = Kr @ u
    return 0


def run(X, Y, M, kernel, la_list, t_list):
    n = X.size(0)
    d = X.size(1)
    dtype = X.dtype
    Xc = X[:M]
    Yc = Y[:M]

    K_M = kernel(Xc, Xc)
    K_M[range(M), range(M)] += 1e-15 * M
    T = torch.cholesky(K_M, upper=True)
    #
    cholT2, cholTt2 = lambda x: tr_solve(x, T), \
                      lambda x: tr_solve(x, T, transpose=True)

    alpha = torch.zeros(M, 1, dtype=dtype)
    alpha2 = torch.zeros(M, 1, dtype=dtype)
    precond = None
    loss = LogisticLoss(kernel, no_keops=True)
    optim = ConjugateGradient()

    for i in range(len(la_list)):
        t = t_list[i]
        la = la_list[i]

        # Compute preconditioner
        if precond is None:
            precond = LogisticPreconditioner(kernel, loss)
        precond.init(Xc, Yc, alpha, la, n, {})
        cholA = precond.invA
        cholAt = precond.invAt
        cholT = precond.invT
        cholTt = precond.invTt
        trmm_out = (alpha2.t() @ T).t()
        #print("TRMM_OUT", trmm_out.T)
        W = loss.ddf(Yc,trmm_out)
        #print("W REAL", W.T)
        A = T.t()
        A = A * W
        A = T @ A
        A /= n
        A[range(M), range(M)] += la
        # print("REAL A (before chol)", A)
        A = torch.cholesky(A, upper=True)
        # np.testing.assert_allclose(A.numpy().T, np.tril(precond.fC), rtol=1e-10)
        #print("REAL T", T)
        #
        # print("MY prec", torch.from_numpy(precond.fC))
        # print("A Diag", precond.dA)
        #print("T Diag", precond.dT)
        cholA2, cholAt2 = lambda x: tr_solve(x, A), \
                          lambda x: tr_solve(x, A, transpose=True)

        # Compute gradient
        lobj = [Y, torch.zeros(n, 1, dtype=dtype)]

        # np.testing.assert_allclose(cholT(alpha).numpy(), cholT2(alpha2).numpy(), rtol=1e-10)

        grad_inner, func_val = loss.knmp_grad(X, Xc, Y, cholT(alpha))
        grad_inner2 = KnmProd(X, Xc, cholT2(alpha2), kernel, aux_grad, lobj, loss)
        # np.testing.assert_allclose(grad_inner.numpy(), grad_inner2.numpy(), rtol=1e-10)

        # np.testing.assert_allclose(cholT(grad_inner), cholT2(grad_inner2), err_msg="cholT", rtol=1e-10)
        # np.testing.assert_allclose(cholTt(grad_inner), cholTt2(grad_inner2), err_msg="cholT transpose", rtol=1e-10)
        # np.testing.assert_allclose(cholA(grad_inner), cholA2(grad_inner2), err_msg="cholA", rtol=1e-10)
        # np.testing.assert_allclose(cholAt(grad_inner), cholAt2(grad_inner2), err_msg="cholA transpose", rtol=1e-10)

        grad_p = cholAt(cholTt(grad_inner) + la * alpha)
        grad_p2 = cholAt2(cholTt2(grad_inner2) + la * alpha2)
        # np.testing.assert_allclose(grad_p.numpy(), grad_p2.numpy(), rtol=1e-10)

        # Compute Hessian
        hess2 = lambda x: cholTt2(KnmProd(X, Xc, cholT2(x), kernel, aux_hess, lobj, loss)) + la * x
        hess_p2 = lambda x: cholAt2(hess2(cholA2(x)))
        hess = lambda x: cholTt(loss.knmp_hess(X, Xc, Y, func_val, cholT(x))) + la * x
        hess_p = lambda x: cholAt(hess(cholA(x)))

        # Perform conjugate gradient with t iterations to obtain approximate newton step
        alpha2 -= cholA(conjgrad(hess_p2, grad_p2, t))
        alpha -= cholA(optim.solve(X0=None, B=grad_p, mmv=hess_p, max_iter=t))
        #np.testing.assert_allclose(alpha.numpy(), alpha2.numpy(), err_msg="ConjGrad Output")

        funval = kernel.mmv(X, Xc, cholT(alpha), opt={"no_keops": True})
        #KnmProd(X, Xc, cholT(alpha), kernel, aux_mmv, lobj, loss)
        #funval = lobj[1]
        error = 100*torch.sum(Y*funval <= 0).to(torch.float64)/Y.shape[0]
        loss_val = torch.mean(loss(Y, funval))
        print("Iteration %d - error %.4f - loss %.4f" % (i, error, loss_val))

    return cholT(alpha)


def KnmProd(X, C, u, kern, l, lobj, loss):
    n = X.size(0)
    blk = 1

    Xs = np.ceil(np.linspace(0, n, blk + 1)).astype(int)
    for i in range(blk):
        X1 = X[Xs[i]:Xs[i + 1], :]
        Kr = kern(X1, C)
        pp = l(loss, lobj, u, Kr, Xs[i], Xs[i+1])
        try:
            p += pp
        except:
            p = pp
    return p


def conjgrad(funA, r, t):
    # initialize parameter
    r0 = r
    p = r
    rsold = torch.sum(r.pow(2))
    n = r.size(0)
    beta = torch.zeros(n, 1, dtype=r.dtype)
    for i in range(t):
        Ap = funA(p)
        a = rsold / (torch.sum(p * Ap))
        aux = a * p
        beta = torch.add(beta, aux)
        aux = - a * Ap
        r = torch.add(r, aux)
        rsnew = torch.sum(r.pow(2))
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return beta

class Test(unittest.TestCase):
    def test_simple_recreated(self):
        from sklearn import datasets
        X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32)).reshape(-1, 1)
        Y[Y == 0] = -1

        # def kernel(X, Y):
        #     return torch.from_numpy(naive_gaussian_kernel(X, Y, sigma=1.0))
        kernel = GaussianKernel(1.0)
        run(X, Y, 100, kernel, [1e-1, 1e-4, 1e-7, 1e-7], [3, 3, 3, 8])