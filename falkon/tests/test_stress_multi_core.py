import os
import pickle
import random
import tempfile
import threading

import numpy as np
import pytest
import torch
from sklearn import datasets

from falkon import FalkonOptions, InCoreFalkon, kernels
from falkon.center_selection import FixedSelector
from falkon.ooc_ops import gpu_lauum
from falkon.utils import decide_cuda


def _ooc_lauum_runner_str(fname_A, fname_out, num_rep, gpu_num):
    run_str = f"""
import numpy
import pickle
import torch
from falkon.ooc_ops.ooc_lauum import gpu_lauum
from falkon import FalkonOptions
from falkon.cuda import initialization

initialization.init(FalkonOptions())

with open('{fname_A}', 'rb') as fh:
    A = pickle.load(fh)

opt = FalkonOptions(compute_arch_speed=False, use_cpu=False, max_gpu_mem=2 * 2**20)
out = []
for rep in range({num_rep}):
    act_up = gpu_lauum(A, upper=True, overwrite=False, opt=opt)
    out.append(act_up)

with open('{fname_out}', 'wb') as fh:
    pickle.dump([o.cpu() for o in out], fh)
    """
    # Save string to temporary file
    py_fname = f"./temp_lauum_runner_gpu{gpu_num}_{random.randint(0, 10000)}.py"
    with open(py_fname, "w") as fh:
        fh.write(run_str)

    os.system(f"CUDA_VISIBLE_DEVICES='{gpu_num}' python {py_fname}")
    os.remove(py_fname)


@pytest.mark.full
@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestStressOocLauum:
    def test_multiple_ooc_lauums(self):
        num_processes = 2
        num_rep = 5
        num_pts = 1536
        num_gpus = torch.cuda.device_count()
        torch.manual_seed(19)
        A = torch.randn(num_pts, num_pts, dtype=torch.float32, device="cpu")
        with tempfile.TemporaryDirectory() as folder:
            A_file = os.path.join(folder, "data_A.pkl")
            out_files = [os.path.join(folder, f"output_{i}.pkl") for i in range(num_processes)]
            # Write data to file
            with open(A_file, "wb") as fh:
                pickle.dump(A, fh)

            threads = []
            for i in range(num_processes):
                t = threading.Thread(
                    target=_ooc_lauum_runner_str,
                    kwargs={
                        "fname_A": A_file,
                        "fname_out": out_files[i],
                        "num_rep": num_rep,
                        "gpu_num": i % num_gpus,
                    },
                    daemon=False,
                )
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Load outputs
            actual = []
            for of in out_files:
                with open(of, "rb") as fh:
                    actual.append(pickle.load(fh))

        # Expected result
        torch.cuda.empty_cache()
        opt = FalkonOptions(compute_arch_speed=False, use_cpu=False, max_gpu_mem=2 * 2**20)
        expected = gpu_lauum(A, upper=True, overwrite=False, opt=opt).cpu()
        # Compare actual vs expected
        wrong = 0
        for i in range(len(actual)):
            for j in range(len(actual[i])):
                try:
                    np.testing.assert_allclose(expected.numpy(), actual[i][j].numpy(), rtol=1e-7)
                except AssertionError:  # noqa: PERF203
                    print(f"Result {j} from process {i} is incorrect")
                    wrong += 1
                    raise
        assert wrong == 0, f"{wrong} results were not equal"


def _runner_str(fname_X, fname_Y, fname_out, num_centers, num_rep, max_iter, gpu_num):
    run_str = f"""
import numpy
import pickle
import torch
from falkon import kernels, FalkonOptions, InCoreFalkon
from falkon.center_selection import FixedSelector

kernel = kernels.GaussianKernel(20.0)
with open('{fname_X}', 'rb') as fh:
    X = pickle.load(fh)
with open('{fname_Y}', 'rb') as fh:
    Y = pickle.load(fh)
X, Y = X.cuda(), Y.cuda()

opt = FalkonOptions(use_cpu=False, keops_active="no", debug=False, never_store_kernel=True,
                    max_gpu_mem=1*2**30, cg_full_gradient_every=2,
                    min_cuda_iter_size_32=0, min_cuda_iter_size_64=0, min_cuda_pc_size_32=0, min_cuda_pc_size_64=0)
out = []
for rep in range({num_rep}):
    center_sel = FixedSelector(X[:{num_centers}])
    flk = InCoreFalkon(
        kernel=kernel, penalty=1e-6, M={num_centers}, seed=10, options=opt, maxiter={max_iter},
        center_selection=center_sel)
    flk.fit(X, Y)
    out.append(flk.predict(X))
with open('{fname_out}', 'wb') as fh:
    pickle.dump([o.cpu() for o in out], fh)
    """
    # Save string to temporary file
    py_fname = f"./temp_runner_gpu{gpu_num}_{random.randint(0, 1000)}.py"
    with open(py_fname, "w") as fh:
        fh.write(run_str)

    os.system(f"CUDA_VISIBLE_DEVICES='{gpu_num}' python {py_fname}")
    os.remove(py_fname)


@pytest.mark.full
@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestStressInCore:
    def test_multiple_falkons(self):
        num_processes = 6
        num_rep = 5
        max_iter = 20
        num_centers = 1000
        num_gpus = torch.cuda.device_count()
        X, Y = datasets.make_regression(50000, 2048, random_state=11)
        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1))

        with tempfile.TemporaryDirectory() as folder:
            X_file = os.path.join(folder, "data_x.pkl")
            Y_file = os.path.join(folder, "data_y.pkl")
            out_files = [os.path.join(folder, f"output_{i}.pkl") for i in range(num_processes)]
            # Write data to file
            with open(X_file, "wb") as fh:
                pickle.dump(X, fh)
            with open(Y_file, "wb") as fh:
                pickle.dump(Y, fh)

            threads = []
            for i in range(num_processes):
                t = threading.Thread(
                    target=_runner_str,
                    kwargs={
                        "fname_X": X_file,
                        "fname_Y": Y_file,
                        "fname_out": out_files[i],
                        "num_rep": num_rep,
                        "max_iter": max_iter,
                        "num_centers": num_centers,
                        "gpu_num": i % num_gpus,
                    },
                    daemon=False,
                )
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Load outputs
            actual = []
            for of in out_files:
                with open(of, "rb") as fh:
                    actual.append(pickle.load(fh))

        # Expected result
        kernel = kernels.GaussianKernel(20.0)
        X, Y = X.cuda(), Y.cuda()
        opt = FalkonOptions(
            use_cpu=False,
            keops_active="no",
            debug=False,
            never_store_kernel=True,
            max_gpu_mem=1 * 2**30,
            cg_full_gradient_every=2,
        )
        center_sel = FixedSelector(X[:num_centers])
        flk = InCoreFalkon(
            kernel=kernel,
            penalty=1e-6,
            M=num_centers,
            seed=10,
            options=opt,
            maxiter=max_iter,
            center_selection=center_sel,
        )
        flk.fit(X, Y)
        expected = flk.predict(X).cpu()
        # Compare actual vs expected
        wrong = 0
        for i in range(len(actual)):
            for j in range(len(actual[i])):
                try:
                    np.testing.assert_allclose(expected.numpy(), actual[i][j].numpy(), rtol=1e-7)
                except AssertionError:  # noqa: PERF203
                    print(f"Result {j} from process {i} is incorrect")
                    wrong += 1
        assert wrong == 0, f"{wrong} results were not equal"
