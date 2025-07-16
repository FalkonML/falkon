import dataclasses

import numpy as np
import pytest
import scipy.linalg.lapack as scll
import torch

from falkon.mmv_ops.utils import CUDA_EXTRA_MM_RAM
from falkon.options import FalkonOptions
from falkon.tests.conftest import memory_checker
from falkon.tests.gen_random import gen_random_pd
from falkon.utils import decide_cuda
from falkon.utils.helpers import sizeof_dtype
from falkon.utils.tensor_helpers import move_tensor

if decide_cuda():
    from falkon.ooc_ops.ooc_potrf import gpu_cholesky


@pytest.fixture(scope="class", params=[4, 1000])
def pd_data(request):
    size = request.param
    return gen_random_pd(size, "float64", F=False, seed=12)


def choose_on_dtype(dtype):
    if dtype == np.float64:
        return scll.dpotrf, 1e-12
    else:
        return scll.spotrf, 1e-5


def run_potrf_test(np_data, dtype, order, opt, input_device, upper, clean, overwrite):
    # Convert pd_data to the appropriate form
    data = np.asarray(np_data, order=order, dtype=dtype, copy=True)
    lapack_fn, rtol = choose_on_dtype(dtype)
    A = move_tensor(torch.from_numpy(data.copy(order="K")), input_device)

    orig_stride = A.stride()
    orig_ptr = A.data_ptr()

    with memory_checker(opt, extra_mem=CUDA_EXTRA_MM_RAM) as new_opt:
        C_gpu = gpu_cholesky(A, upper=upper, clean=clean, overwrite=overwrite, opt=new_opt)

    assert orig_stride == C_gpu.stride(), "gpu_potrf modified matrix stride."
    if overwrite:
        assert orig_ptr == C_gpu.data_ptr(), "Data-pointer changed although overwrite is True."

    C_cpu = lapack_fn(data, lower=int(not upper), clean=int(clean), overwrite_a=int(overwrite))[0]
    np.testing.assert_allclose(C_cpu, C_gpu.cpu().numpy(), rtol=rtol, verbose=True)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.full())])
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("overwrite", [True, False])
class TestInCorePyTest:
    basic_options = FalkonOptions(debug=True, chol_force_in_core=True)

    @pytest.mark.parametrize("clean", [True, False])
    @pytest.mark.parametrize("order", ["F", "C"])
    @pytest.mark.parametrize("input_device", ["cpu", "cuda:0"])
    def test_in_core(self, pd_data, dtype, order, upper, clean, overwrite, input_device):
        run_potrf_test(
            pd_data,
            dtype=dtype,
            order=order,
            upper=upper,
            clean=clean,
            overwrite=overwrite,
            input_device=input_device,
            opt=self.basic_options,
        )

    @pytest.mark.full
    @pytest.mark.parametrize("clean,order,input_device", [pytest.param(False, "F", "cpu")])
    def test_ic_mem(self, pd_data, dtype, order, upper, clean, overwrite, input_device):
        if input_device.startswith("cuda"):
            max_mem = 2000
        else:
            # 1600 is needed!
            max_mem = max(1600, pd_data.shape[0] * pd_data.shape[1] * sizeof_dtype(dtype) * 1.5)
        opt = dataclasses.replace(self.basic_options, max_gpu_mem=max_mem)

        run_potrf_test(
            pd_data,
            dtype=dtype,
            order=order,
            upper=upper,
            clean=clean,
            overwrite=overwrite,
            input_device=input_device,
            opt=opt,
        )

    @pytest.mark.full
    @pytest.mark.parametrize("clean,order,input_device", [pytest.param(False, "F", "cpu")])
    def test_ic_mem_fail(self, pd_data, dtype, order, upper, clean, overwrite, input_device):
        if input_device.startswith("cuda"):
            max_mem = 10
        else:
            max_mem = pd_data.shape[0]
        opt = dataclasses.replace(self.basic_options, max_gpu_mem=max_mem)
        # Will raise due to insufficient memory
        with pytest.raises(RuntimeError, match="Cannot run in-core POTRF but `chol_force_in_core` was specified."):
            run_potrf_test(
                pd_data,
                dtype=dtype,
                order=order,
                upper=upper,
                clean=clean,
                overwrite=overwrite,
                input_device=input_device,
                opt=opt,
            )


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
@pytest.mark.parametrize("dtype", [np.float32, pytest.param(np.float64, marks=pytest.mark.full())])
@pytest.mark.parametrize("overwrite", [True, False])
class TestOutOfCorePyTest:
    basic_options = FalkonOptions(debug=True, chol_force_ooc=True)

    def test_start_cuda_fail(self, pd_data, dtype, overwrite):
        # Cannot run OOC-POTRF on CUDA matrices (only IC-POTRF allowed)
        with pytest.raises(ValueError, match="Cannot run out-of-core POTRF on CUDA"):
            run_potrf_test(
                pd_data,
                dtype=dtype,
                order="F",
                upper=False,
                clean=False,
                overwrite=overwrite,
                input_device="cuda:0",
                opt=self.basic_options,
            )

    @pytest.mark.parametrize("clean", [True, False])
    @pytest.mark.parametrize(
        "order,upper",
        [
            pytest.param(
                "F",
                True,
                marks=[
                    pytest.mark.xfail(strict=True),
                ],
            ),  # Upper-F not possible
            pytest.param("C", True),
            pytest.param("F", False),
            pytest.param(
                "C",
                False,
                marks=[
                    pytest.mark.xfail(strict=True),
                ],
            ),  # Lower-C not possible
        ],
    )
    def test_ooc(self, pd_data, dtype, order, upper, clean, overwrite):
        run_potrf_test(
            pd_data,
            dtype=dtype,
            order=order,
            upper=upper,
            clean=clean,
            overwrite=overwrite,
            input_device="cpu",
            opt=self.basic_options,
        )

    @pytest.mark.parametrize(
        "clean,order,upper",
        [
            pytest.param(False, "C", True),
            pytest.param(True, "F", False),
        ],
    )
    def test_ooc_mem(self, pd_data, dtype, order, upper, clean, overwrite):
        # 1600 is the minimum memory the fn seems to use (even for the 4x4 data)
        max_mem = max(pd_data.shape[0] * sizeof_dtype(dtype) * 1000, 1600)
        opt = dataclasses.replace(self.basic_options, max_gpu_mem=max_mem)
        run_potrf_test(
            pd_data,
            dtype=dtype,
            order=order,
            upper=upper,
            clean=clean,
            overwrite=overwrite,
            input_device="cpu",
            opt=opt,
        )


if __name__ == "__main__":
    pytest.main()
