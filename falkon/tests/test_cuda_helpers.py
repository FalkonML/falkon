import numpy as np
import pytest
import torch

from falkon.tests.gen_random import gen_random
from falkon.utils import decide_cuda

if decide_cuda():
    from falkon.utils import cuda_helpers


@pytest.fixture(params=[True, False], ids=["F", "C"])
def host_mat(request):
    return torch.from_numpy(gen_random(20, 20, 'float32', F=request.param))


@pytest.fixture(params=[True, False], ids=["F", "C"])
def dev_mat(request):
    return torch.from_numpy(gen_random(20, 20, 'float32', F=request.param)).cuda()


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
def test_copy_to_device_noorder(host_mat, dev_mat):
    copy_out = cuda_helpers.copy_to_device_noorder(10, 10, host_mat, 5, 5, dev_mat, 5, 5)

    assert copy_out.data_ptr() == dev_mat[5,5].data_ptr()
    np.testing.assert_allclose(host_mat[5:15,5:15].numpy(), dev_mat[5:15,5:15].cpu().numpy())
