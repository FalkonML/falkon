import torch


def cholesky(M, upper=False, check_errors=True):
    if upper:
        U, info = torch.linalg.cholesky_ex(M.transpose(-2, -1).conj())
        if check_errors:
            if info > 0:
                raise RuntimeError(f"Cholesky failed on row {info}")
        return U.transpose(-2, -1).conj()
    else:
        L, info = torch.linalg.cholesky_ex(M, check_errors=False)
        if check_errors:
            if info > 0:
                raise RuntimeError(f"Cholesky failed on row {info}")
        return L


def jittering_cholesky(mat, upper=False):
    eye = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
    epsilons = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    last_exception = None
    for eps in epsilons:
        try:
            return cholesky(mat + eye * eps, upper=upper, check_errors=True)
        except RuntimeError as e:  # noqa: PERF203
            last_exception = e
    raise last_exception
