import pytest

import torch
import numpy as np
import scipy.linalg

from falkon.c_ext import trtri as my_trtri

num_points = 10

def test_trtri():
    w = torch.randn(10, 10)
    w = w @ w.T
    z = torch.cholesky(w, upper=True)

    expected = scipy.linalg.lapack.strtri(z.numpy().copy(), lower=0, unitdiag=0, overwrite_c=0)[0]
    actual = my_trtri(z.clone().cuda(), lower=False, unitdiag=False)
    actual = actual.cpu().numpy()
    np.testing.assert_allclose(expected, actual)


