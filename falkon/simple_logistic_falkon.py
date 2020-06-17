from scipy.spatial.distance import cdist
import torch
import numpy as np

import sys
sys.path.append("/home/giacomo/unige/falkon/FALKON")


def naive_gaussian_kernel(X1, X2, sigma):
    pairwise_dists = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


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


def aux_fun(lobj, u, Kr, a, b):
    p = Kr @ u
    lobj[1] = p[a:b]
    return f(lobj[0][a:b], p).sum() / Kr.shape[0]


def aux_grad(lobj, u, Kr, a, b):
    p = Kr @ u
    lobj[1][a:b] = p
    _df = df(lobj[0][a:b], p)
    return (_df.t() @ Kr).t() / Kr.shape[0]


def aux_hess(lobj, u, Kr, a, b):
    p = Kr @ u
    return ((ddf(lobj[0][a:b], lobj[1][a:b]) * p).t() @ Kr).t() / Kr.shape[0]


def aux_mmv(lobj, u, Kr, a, b):
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

    cholT, cholTt = lambda x: tr_solve(x, T), \
                   lambda x: tr_solve(x, T, transpose=True)

    alpha = torch.zeros(M, 1, dtype=dtype)

    for i in range(len(la_list)):
        t = t_list[i]
        la = la_list[i]

        # Compute preconditioner

        # if precond is None:
        #     precond = falkon.precond.LogisticPreconditioner(
        #         kernel, self.loss, self.extra_opt_)
        # precond.init(ny_X, ny_Y, beta, penalty, self.extra_opt_)
        W = ddf(Yc, (alpha.t() @ T).t())
        A = T.t()
        A = A * W
        A = T @ A
        A /= n
        A[range(M), range(M)] += la
        A = torch.cholesky(A, upper=True)
        cholA, cholAt = lambda x: tr_solve(x, A), \
                        lambda x: tr_solve(x, A, transpose=True)

        # Compute gradient
        lobj = [Y, torch.zeros(n, 1, dtype=dtype)]
        grad_inner = KnmProd(X, Xc, cholT(alpha), kernel, aux_grad, lobj)
        grad_p = cholAt(cholTt(grad_inner) + la * alpha)

        # Compute Hessian
        hess = lambda x: cholTt(KnmProd(X, Xc, cholT(x), kernel, aux_hess, lobj)) + la * x
        hess_p = lambda x: cholAt(hess(cholA(x)))

        # Perform conjugate gradient with t iterations to obtain approximate newton step
        alpha -= cholA(conjgrad(hess_p, grad_p, t))

        KnmProd(X, Xc, cholT(alpha), kernel, aux_mmv, lobj)
        funval = lobj[1]
        error = 100*torch.sum(Y*funval <= 0).to(torch.float64)/Y.shape[0]
        loss = torch.mean(f(Y,funval))
        print("Iteration %d - error %.4f - loss %.4f" % (i, error, loss))

    return cholT(alpha)


def KnmProd(X, C, u, kern, l, lobj):
    n = X.size(0)
    blk = 1

    Xs = np.ceil(np.linspace(0, n, blk + 1)).astype(int)
    for i in range(blk):
        X1 = X[Xs[i]:Xs[i + 1], :]
        Kr = kern(X1, C)
        pp = l(lobj, u, Kr, Xs[i], Xs[i+1])
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


if __name__ == "__main__":
    from sklearn import datasets
    X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
    X = torch.from_numpy(X.astype(np.float64))
    Y = torch.from_numpy(Y.astype(np.float64)).reshape(-1, 1)
    Y[Y == 0] = -1

    def kernel(X, Y):
        return torch.from_numpy(naive_gaussian_kernel(X, Y, sigma=1.0))
    run(X, Y, 100, kernel, [1e-1, 1e-4, 1e-7, 1e-7], [3, 3, 3, 8])