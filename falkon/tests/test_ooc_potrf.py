import numpy as np
import pytest
import scipy.linalg.lapack as scll
import torch
from falkon.tests.conftest import memory_checker

from falkon.ooc_ops.options import CholeskyOptions
from falkon.tests.helpers import gen_random_pd
from falkon.utils import decide_cuda
from falkon.utils.helpers import sizeof_dtype

if decide_cuda({}):
    from falkon.ooc_ops.ooc_potrf import gpu_cholesky


@pytest.fixture(scope="class", params=[4, 4000])
def pd_data(request):
    size = request.param
    return gen_random_pd(size, 'float64', F=False, seed=12)


def choose_on_dtype(dtype):
    if dtype == np.float64:
        return scll.dpotrf, 1e-12
    else:
        return scll.spotrf, 1e-5


def run_potrf_test(np_data, dtype, order, opt, start_cuda, upper, clean, overwrite, check_mem):
    # Convert pd_data to the appropriate form
    data = np.copy(np_data, order=order).astype(dtype, copy=False)
    lapack_fn, rtol = choose_on_dtype(dtype)
    A = torch.from_numpy(data.copy(order="K"))
    if start_cuda:
        A = A.cuda()

    orig_stride = A.stride()
    orig_ptr = A.data_ptr()

    with memory_checker(opt) as new_opt:
        C_gpu = gpu_cholesky(A, upper=upper, clean=clean, overwrite=overwrite, opt=new_opt)

    assert orig_stride == C_gpu.stride(), "gpu_potrf modified matrix stride."
    if overwrite:
        assert orig_ptr == C_gpu.data_ptr(), "Data-pointer changed although overwrite is True."

    C_cpu = lapack_fn(data, lower=int(not upper), clean=int(clean), overwrite_a=int(overwrite))[0]
    np.testing.assert_allclose(C_cpu, C_gpu.cpu().numpy(), rtol=rtol, verbose=True)


@pytest.mark.skipif(not decide_cuda({}), reason="No GPU found.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("overwrite", [True, False])
class TestInCorePyTest:
    basic_options = {
        'debug': True,
        'cholesky_opt': CholeskyOptions(
            chol_force_in_core=True,
        ),
    }

    @pytest.mark.parametrize("clean,order,start_cuda", [
        pytest.param(True, "F", True, marks=[pytest.mark.xfail(strict=True), ]),  # Cannot clean with CUDA input
        pytest.param(True, "F", False),
        pytest.param(True, "C", True, marks=[pytest.mark.xfail(strict=True), ]),  # Cannot clean with CUDA input
        pytest.param(True, "C", False),
        pytest.param(False, "F", True),
        pytest.param(False, "F", False),
        pytest.param(False, "C", True),
        pytest.param(False, "C", False),
    ])
    def test_in_core(self, pd_data, dtype, order, upper, clean, overwrite, start_cuda):
        run_potrf_test(pd_data, dtype=dtype, order=order, upper=upper, clean=clean,
                       overwrite=overwrite, start_cuda=start_cuda, opt=self.basic_options,
                       check_mem=False)

    @pytest.mark.parametrize("clean,order,start_cuda", [
        pytest.param(False, "F", False),
    ])
    def test_ic_mem(self, pd_data, dtype, order, upper, clean, overwrite, start_cuda):
        if start_cuda:
            max_mem = 2000
        else:
            # 1600 is needed!
            max_mem = max(1600, pd_data.shape[0] * pd_data.shape[1] * sizeof_dtype(dtype) * 1.5)
        opt = dict(self.basic_options)
        opt['max_gpu_mem'] = max_mem

        run_potrf_test(pd_data, dtype=dtype, order=order, upper=upper, clean=clean,
                       overwrite=overwrite, start_cuda=start_cuda, opt=opt,
                       check_mem=True)

    @pytest.mark.parametrize("clean,order,start_cuda", [
        pytest.param(False, "F", False, marks=pytest.mark.xfail(
            reason="Insufficient GPU memory for test to pass.", strict=True,
            raises=RuntimeError)),
    ])
    def test_ic_mem_fail(self, pd_data, dtype, order, upper, clean, overwrite, start_cuda):
        if start_cuda:
            max_mem = 10
        else:
            max_mem = pd_data.shape[0]
        opt = dict(self.basic_options)
        opt['max_gpu_mem'] = max_mem

        run_potrf_test(pd_data, dtype=dtype, order=order, upper=upper, clean=clean,
                       overwrite=overwrite, start_cuda=start_cuda, opt=opt,
                       check_mem=True)


@pytest.mark.skipif(not decide_cuda({}), reason="No GPU found.")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("overwrite", [True, False])
class TestOutOfCorePyTest():
    basic_options = {
        'debug': True,
        'cholesky_opt': CholeskyOptions(
            chol_force_ooc=True,
        ),
    }

    @pytest.mark.xfail(raises=ValueError, strict=True)
    def test_start_cuda_fail(self, pd_data, dtype, overwrite):
        run_potrf_test(pd_data, dtype=dtype, order="F", upper=False, clean=False,
                       overwrite=overwrite, start_cuda=True, opt=self.basic_options,
                       check_mem=False)

    @pytest.mark.parametrize("clean,order,upper", [
        pytest.param(True, "F", True, marks=[pytest.mark.xfail(strict=True), ]), # Upper-F not possible
        pytest.param(False, "F", True, marks=[pytest.mark.xfail(strict=True), ]), # Upper-F not possible
        pytest.param(True, "C", True),
        pytest.param(False, "C", True),
        pytest.param(True, "F", False),
        pytest.param(False, "F", False),
        pytest.param(True, "C", False, marks=[pytest.mark.xfail(strict=True), ]), # Lower-C not possible
        pytest.param(False, "C", False, marks=[pytest.mark.xfail(strict=True), ]), # Lower-C not possible
    ])
    def test_ooc(self, pd_data, dtype, order, upper, clean, overwrite):
        run_potrf_test(pd_data, dtype=dtype, order=order, upper=upper, clean=clean,
                       overwrite=overwrite, start_cuda=False, opt=self.basic_options,
                       check_mem=False)

    @pytest.mark.parametrize("clean,order,upper", [
        pytest.param(False, "C", True),
        pytest.param(True, "F", False),
    ])
    def test_ooc_mem(self, pd_data, dtype, order, upper, clean, overwrite):
        # 1600 is the minimum memory the fn seems to use (even for the 4x4 data)
        max_mem = max(pd_data.shape[0] * sizeof_dtype(dtype) * 1000, 1600)
        opt = dict(self.basic_options)
        opt['max_gpu_mem'] = max_mem
        run_potrf_test(pd_data, dtype=dtype, order=order, upper=upper, clean=clean,
                       overwrite=overwrite, start_cuda=False, opt=opt,
                       check_mem=True)
