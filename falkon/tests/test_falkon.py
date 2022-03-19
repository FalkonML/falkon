import os
import pickle
import random
import tempfile
import threading

import numpy as np
import pytest
import torch
from sklearn import datasets

from falkon.models.incore_falkon import InCoreFalkon
from falkon.center_selection import FixedSelector
from falkon import Falkon, kernels
from falkon.options import FalkonOptions
from falkon.utils import decide_cuda


@pytest.fixture
def cls_data():
    X, Y = datasets.make_classification(1000, 10, n_classes=2, random_state=11)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64).reshape(-1, 1)
    Y[Y == 0] = -1
    return torch.from_numpy(X), torch.from_numpy(Y)


@pytest.fixture
def multicls_data():
    X, Y = datasets.make_classification(2000, 30, n_classes=20, n_informative=6, random_state=11)
    X = X.astype(np.float64)
    eye = np.eye(20, dtype=np.float64)
    Y = eye[Y.astype(np.int32), :]
    return torch.from_numpy(X), torch.from_numpy(Y)


@pytest.fixture
def reg_data():
    X, Y = datasets.make_regression(1000, 30, random_state=11)
    num_train = 800
    X = X.astype(np.float64)
    Y = Y.astype(np.float64).reshape(-1, 1)
    return (torch.from_numpy(X[:num_train]), torch.from_numpy(Y[:num_train]),
            torch.from_numpy(X[num_train:]), torch.from_numpy(Y[num_train:]))


class TestFalkon:
    def test_no_opt(self):
        kernel = kernels.GaussianKernel(2.0)
        Falkon(
            kernel=kernel, penalty=1e-6, M=500, center_selection='uniform', options=None,
        )

    def test_classif(self, cls_data):
        X, Y = cls_data
        kernel = kernels.GaussianKernel(2.0)
        torch.manual_seed(13)
        np.random.seed(13)

        def error_fn(t, p):
            return 100 * torch.sum(t * p <= 0).to(torch.float32) / t.shape[0], "c-err"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)
        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt, maxiter=10)
        flk.fit(X, Y)
        preds = flk.predict(X)
        err = error_fn(preds, Y)[0]
        assert err < 5

    def test_multiclass(self, multicls_data):
        X, Y = multicls_data
        kernel = kernels.GaussianKernel(10.0)

        def error_fn(t, p):
            t = torch.argmax(t, dim=1)
            p = torch.argmax(p, dim=1)
            return float(
                torch.mean((t.reshape(-1, ) != p.reshape(-1, )).to(torch.float64))), "multic-err"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)
        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt, maxiter=10)
        flk.fit(X, Y)
        preds = flk.predict(X)
        err = error_fn(preds, Y)[0]
        assert err < 0.23

    def test_regression(self, reg_data):
        Xtr, Ytr, Xts, Yts = reg_data
        kernel = kernels.GaussianKernel(20.0)

        def error_fn(t, p):
            return torch.sqrt(torch.mean((t - p) ** 2)).item(), "RMSE"

        opt = FalkonOptions(use_cpu=True, keops_active="no", debug=True)
        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=Xtr.shape[0] // 2, seed=10,
            options=opt, maxiter=10)
        flk.fit(Xtr, Ytr, Xts=Xts, Yts=Yts)

        assert flk.predict(Xts).shape == (Yts.shape[0], 1)
        ts_err = error_fn(flk.predict(Xts), Yts)[0]
        tr_err = error_fn(flk.predict(Xtr), Ytr)[0]
        assert tr_err < ts_err
        assert ts_err < 2.5

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_cuda_predict(self, reg_data):
        Xtr, Ytr, Xts, Yts = reg_data
        kernel = kernels.GaussianKernel(20.0)

        def error_fn(t, p):
            return torch.sqrt(torch.mean((t - p) ** 2)).item(), "RMSE"

        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=True,
                            min_cuda_pc_size_64=1, min_cuda_iter_size_64=1)

        flk = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt,
            error_fn=error_fn)
        flk.fit(Xtr, Ytr, Xts=Xts, Yts=Yts)
        flk.to("cuda:0")

        cuda_ts_preds = flk.predict(Xts.to("cuda:0"))
        cuda_tr_preds = flk.predict(Xtr.to("cuda:0"))
        assert cuda_ts_preds.device.type == "cuda"
        assert cuda_ts_preds.shape == (Yts.shape[0], 1)
        ts_err = error_fn(cuda_ts_preds.cpu(), Yts)[0]
        tr_err = error_fn(cuda_tr_preds.cpu(), Ytr)[0]
        assert tr_err < ts_err
        assert ts_err < 2.5

    @pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
    def test_compare_cuda_cpu(self, reg_data):
        Xtr, Ytr, Xts, Yts = reg_data
        kernel = kernels.GaussianKernel(20.0)

        def error_fn(t, p):
            return torch.sqrt(torch.mean((t - p) ** 2)).item(), "RMSE"

        opt_cpu = FalkonOptions(use_cpu=True, keops_active="no", debug=True)
        flk_cpu = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt_cpu, maxiter=10, error_fn=error_fn)
        flk_cpu.fit(Xtr, Ytr, Xts=Xts, Yts=Yts)
        opt_gpu = FalkonOptions(use_cpu=False, keops_active="no", debug=True)
        flk_gpu = Falkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt_gpu, maxiter=10, error_fn=error_fn)
        flk_gpu.fit(Xtr, Ytr, Xts=Xts, Yts=Yts)

        np.testing.assert_allclose(flk_cpu.alpha_.numpy(), flk_gpu.alpha_.numpy())


class TestWeightedFalkon:
    @pytest.mark.parametrize("cuda_usage", [
        pytest.param("incore",
                     marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")]),
        pytest.param("mixed",
                     marks=[pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")]),
        "cpu_only",
    ])
    def test_classif(self, cls_data, cuda_usage):
        X, Y = cls_data
        if cuda_usage == "incore":
            X, Y = X.cuda(), Y.cuda()
            flk_cls = InCoreFalkon
        else:
            flk_cls = Falkon
        kernel = kernels.GaussianKernel(2.0)

        def error_fn(t, p):
            return 100 * torch.sum(t * p <= 0).to(torch.float32) / t.shape[0], "c-err"

        def weight_fn(y, x, indices):
            weight = torch.empty_like(y)
            weight[y == 1] = 1
            weight[y == -1] = 2
            return weight

        opt = FalkonOptions(use_cpu=cuda_usage == "cpu_only", keops_active="no", debug=False)

        flk_weight = flk_cls(kernel=kernel, penalty=1e-6, M=500, seed=10, options=opt,
                             error_fn=error_fn, weight_fn=weight_fn)
        flk_weight.fit(X, Y)
        preds_weight = flk_weight.predict(X)
        preds_weight_m1 = preds_weight[Y == -1]
        preds_weight_p1 = preds_weight[Y == 1]
        err_weight_m1 = error_fn(preds_weight_m1, Y[Y == -1])[0]
        err_weight_p1 = error_fn(preds_weight_p1, Y[Y == 1])[0]

        flk = flk_cls(kernel=kernel, penalty=1e-6, M=500, seed=10, options=opt,
                      error_fn=error_fn, weight_fn=None)
        flk.fit(X, Y)
        preds = flk.predict(X)
        preds_m1 = preds[Y == -1]
        preds_p1 = preds[Y == 1]
        err_m1 = error_fn(preds_m1, Y[Y == -1])[0]
        err_p1 = error_fn(preds_p1, Y[Y == 1])[0]

        print("Weighted errors: -1 (%f) +1 (%f) -- Normal errors: -1 (%f) +1 (%f)" % (
            err_weight_m1, err_weight_p1, err_m1, err_p1))

        assert err_weight_m1 < err_m1, "Error of weighted class is higher than without weighting"
        assert err_weight_p1 >= err_p1, "Error of unweighted class is lower than in flk with no weights"


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestIncoreFalkon:
    def test_fails_cpu_tensors(self, cls_data):
        X, Y = cls_data
        kernel = kernels.GaussianKernel(2.0)

        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=True)

        flk = InCoreFalkon(
            kernel=kernel, penalty=1e-6, M=500, seed=10,
            options=opt)
        with pytest.raises(ValueError):
            flk.fit(X, Y)
        flk.fit(X.cuda(), Y.cuda())
        with pytest.raises(ValueError):
            flk.predict(X)

    def test_classif(self, cls_data):
        X, Y = cls_data
        Xc = X.cuda()
        Yc = Y.cuda()
        kernel = kernels.GaussianKernel(2.0)
        torch.manual_seed(13)
        np.random.seed(13)

        def error_fn(t, p):
            return 100 * torch.sum(t * p <= 0).to(torch.float32) / t.shape[0], "c-err"

        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=True)
        M = 500
        flkc = InCoreFalkon(
            kernel=kernel, penalty=1e-6, M=M, seed=10,
            options=opt, maxiter=20,
            error_fn=error_fn)
        flkc.fit(Xc, Yc)

        cpreds = flkc.predict(Xc)
        assert cpreds.device == Xc.device
        err = error_fn(cpreds, Yc)[0]
        assert err < 5


def _runner_str(fname_X, fname_Y, fname_out, num_centers, num_rep, max_iter, gpu_num):
    run_str = f"""
    import pickle
    import torch
    from falkon import kernels, FalkonOptions, InCoreFalkon
    from falkon.center_selection import FixedSelector

    num_rep = 5
    kernel = kernels.GaussianKernel(20.0)
    with open({fname_X}, 'rb') as fh:
        X = pickle.load(fh)
    with open({fname_Y}, 'rb') as fh:
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
    with open({fname_out}, 'wb') as fh:
        pickle.dump([o.cpu() for o in out], fh)
    """
    # Save string to temporary file
    py_fname = f"./temp_runner_gpu{gpu_num}_{random.randint(0, 1000)}.py"
    with open(py_fname, 'w') as fh:
        fh.write(run_str)

    os.system(f"CUDA_VISIBLE_DEVICES='{gpu_num}' python {py_fname}")


@pytest.mark.skipif(not decide_cuda(), reason="No GPU found.")
class TestStressInCore:
    def _test_multiple_falkons(self):
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
                                         'gpu_num': num_gpus % (i + 1)
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
        opt = FalkonOptions(use_cpu=False, keops_active="no", debug=False,
                            max_gpu_mem=1 * 2**30, cg_full_gradient_every=10)
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
            except:  # noqa E722
                wrong += 1
        assert wrong == 0, "%d results were not equal" % (wrong)
