import os
import pickle
import random
import tempfile
import threading

import pytest
import torch
import numpy as np

from falkon.center_selection import FixedSelector

from falkon import kernels, FalkonOptions, InCoreFalkon
from sklearn import datasets

from falkon.utils import decide_cuda


def _runner_str(fname_X, fname_Y, fname_out, num_centers, num_rep, max_iter, gpu_num):
    run_str = f"""
    import pickle
    import torch
    from falkon import kernels, FalkonOptions, InCoreFalkon
    from falkon.center_selection import FixedSelector
    
    num_rep = 5
    kernel = kernels.GaussianKernel(20.0)
    with open('{fname_X}', 'rb') as fh:
        X = pickle.load(fh)
    with open('{fname_Y}', 'rb') as fh:
        Y = pickle.load(fh)
    X, Y = X.cuda(), Y.cuda()
    
    opt = FalkonOptions(use_cpu=False, keops_active="no", debug=False, #never_store_kernel=True,
                        max_gpu_mem=1*2**30, cg_full_gradient_every=10,
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
    with open(py_fname, 'w') as fh:
        fh.write(run_str)

    os.system(f"CUDA_VISIBLE_DEVICES='{gpu_num}' python {py_fname}")
    os.remove(py_fname)


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestStressInCore:
    def test_multiple_falkons(self):
        num_processes = 6
        num_rep = 5
        max_iter = 15
        num_gpus = torch.cuda.device_count()
        X, Y = datasets.make_regression(50000, 2048, random_state=11)
        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32).reshape(-1, 1))

        with tempfile.TemporaryDirectory() as folder:
            X_file = os.path.join(folder, "data_x.pkl")
            Y_file = os.path.join(folder, "data_y.pkl")
            out_files = [
                os.path.join(folder, f"output_{i}.pkl") for i in range(num_processes)
            ]
            # Write data to file
            with open(X_file, 'wb') as fh:
                pickle.dump(X, fh)
            with open(Y_file, 'wb') as fh:
                pickle.dump(Y, fh)

            threads = []
            for i in range(num_processes):
                t = threading.Thread(target=_runner_str,
                                     kwargs={
                                         'fname_X': X_file,
                                         'fname_Y': Y_file,
                                         'fname_out': out_files[i],
                                         'num_rep': num_rep,
                                         'max_iter': max_iter,
                                         'gpu_num': i % num_gpus
                                     },
                                     daemon=False)
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Expected result
        kernel = kernels.GaussianKernel(20.0)
        X, Y = X.cuda(), Y.cuda()
        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=False, #never_store_kernel=True,
                            max_gpu_mem=1*2**30, cg_full_gradient_every=10)
        center_sel = FixedSelector(X[:2000])
        flk = InCoreFalkon(
            kernel=kernel, penalty=1e-6, M=2000, seed=10, options=opt,
            maxiter=15, center_selection=center_sel)
        flk.fit(X, Y)
        expected = flk.predict(X).cpu()
        # Load outputs
        actual = []
        for of in out_files:
            with open(of, 'rb') as fh:
                actual.append(pickle.load(fh))
        # Compare actual vs expected
        wrong = 0
        for i in range(len(actual)):
            try:
                np.testing.assert_allclose(expected.numpy(), actual[i].numpy(), rtol=1e-7)
            except:
                wrong += 1
        assert wrong == 0, "%d results were not equal" % (wrong)