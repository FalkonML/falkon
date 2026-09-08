import time

import torch
from fast_krr.models import FullKRR
from fast_krr.opts import ASkotchV2
from tqdm import trange


class ASkotchWrapper:
    def __init__(
        self,
        block_size,
        precond_params,
        kernel_type,
        kernel_sigma,
        kernel_nu,
        unsc_lam,
        task,
        num_iter,
        device,
        log_every=1
    ):
        self.block_size = block_size
        self.precond_params = precond_params
        self.kernel_type = kernel_type
        self.kernel_sigma = kernel_sigma
        self.kernel_nu = kernel_nu
        self.unsc_lam = unsc_lam
        self.task = task
        self.num_iter = num_iter
        self.device = device
        self.log_every = log_every
        self.fit_times_ = []

        self.error_fn = None
        self.model = None
        self.mc_weights = None

    def get_median_sigma(self, data, num_samples=10000):
        sub_data = data[:num_samples]
        return torch.median(torch.pdist(sub_data))

    def get_kernel_params(self, X):
        sigma = self.kernel_sigma
        if sigma < 0:
            sigma = self.get_median_sigma(X).item()
            print(f"kernel sigma chosen with median heuristic: {sigma}")
        if self.kernel_type == 'gaussian':
            return {'type' : 'rbf', 'sigma': sigma}
        elif self.kernel_type == 'laplacian':
            return {'type' : 'l1_laplace', 'sigma': sigma}
        elif self.kernel_type == 'matern':
            return {'type' : 'matern', 'sigma' : sigma, 'nu' : self.kernel_nu}
        else:
            raise ValueError(f"Kernel {self.kernel_type} not valid for ASkotch")

    def inter_epoch_cback(self, Xts, Yts):
        if self.task != "mc-classification":
            print("Running test-set predictions...", flush=True)
            pred_start_time = time.time()
            preds = self.predict(Xts)
            pred_elapsed = time.time() - pred_start_time
            print(f"ASkotch epoch {len(self.fit_times_) - 1}:")
            print(f"\telapsed: {self.fit_times_[-1]:.2f}s - predictions in {pred_elapsed:.2f}s", flush=True)
            if self.error_fn is not None:
                test_err, test_err_name = self.error_fn(Yts.unsqueeze(1), preds)
                print(f"\ttest {test_err_name}: {test_err:9.6f}", flush=True)
            print()
        else:
            print(f"ASkotch epoch {len(self.fit_times_) - 1}:")
            print(f"\telapsed: {self.fit_times_[-1]:.2f}s", flush=True)
            print()

    def get_block_size(self, Xtr):
        block_size = self.block_size
        if block_size <= 0:
            block_size = Xtr.shape[0] // 100
        return block_size

    def init_model(self, Xtr, Ytr, Xts, Yts):
        if Ytr.dim() > 1:  # assume Yts has same ndims
            assert Ytr.shape[1] == 1, "Unsupported multiple targets with ASkotch"
            Ytr = Ytr.squeeze(1)
            Yts = Yts.squeeze(1)
        block_size = self.get_block_size(Xtr)
        w0 = torch.zeros((Xtr.shape[0], ), device=self.device)
        krr = FullKRR(
            Xtr, Ytr, Xts, Yts, kernel_params=self.get_kernel_params(Xtr),
            Ktr_needed=True, lambd=self.unsc_lam * Xtr.shape[0], task=self.task, w0=w0,
            device=self.device
        )
        self.model = ASkotchV2(model=krr, block_sz=block_size, precond_params=self.precond_params)
        return None

    def _fit_internal(self, Xtr, Ytr, Xts, Yts, initial_time=0.0):
        if Ytr.dim() > 1:  # assume Yts has same ndims
            assert Ytr.shape[1] == 1, "Unsupported multiple targets with ASkotch"
            Ytr = Ytr.squeeze(1)
            Yts = Yts.squeeze(1)
        if self.model is None:
            self.init_model(Xtr, Ytr, Xts, Yts)
        assert self.model is not None
        self.fit_times_ = [initial_time]
        block_size = self.get_block_size(Xtr)
        cback_every = Xtr.shape[0] // block_size
        t_start = time.time()
        for i in trange(1, self.num_iter + 1, desc="Optimization progress"):
            self.model.step()
            if (i % cback_every) == 0:
                t_elapsed = time.time() - t_start
                self.fit_times_.append(self.fit_times_[-1] + t_elapsed)
                # Callback excluded from timings
                self.inter_epoch_cback(Xts, Yts)
                # resume timings
                t_start = time.time()
        # Final time
        t_elapsed = time.time() - t_start
        self.fit_times_.append(self.fit_times_[-1] + t_elapsed)

    def fit(self, Xtr, Ytr, Xts, Yts):
        if self.task == "mc-classification":
            self.mc_weights = []
            tot_time = 0.0
            num_classes = Ytr.shape[1]
            assert num_classes > 1
            for target_class in range(num_classes):
                print(f"Training model for target class {target_class} (out of {num_classes})")
                bin_ytr = Ytr.argmax(-1).to(Xtr.device, Xtr.dtype)
                bin_yts = Yts.argmax(-1).to(Xtr.device, Xtr.dtype)
                bin_ytr[bin_ytr != target_class] = -1.0
                bin_ytr[bin_ytr == target_class] = 1.0
                bin_yts[bin_yts != target_class] = -1.0
                bin_yts[bin_yts == target_class] = 1.0
                self.model = None
                self.init_model(Xtr, bin_ytr, Xts, bin_yts)
                assert self.model is not None
                self._fit_internal(Xtr, bin_ytr, Xts, bin_yts, initial_time=tot_time)
                self.mc_weights.append(self.model.model.w.detach().cpu())
                tot_time = self.fit_times_[-1]
        else:
            self._fit_internal(Xtr, Ytr, Xts, Yts)

    def predict(self, Xtst):
        if self.model is None:
            raise ValueError("predict called before fit")
        kern_fn = self.model.model._get_kernel_fn()
        K_pred = kern_fn(Xtst, self.model.model.x, False)
        if self.task == "mc-classification":
            assert self.mc_weights is not None
            mc_preds = []
            for target_class, w in enumerate(self.mc_weights):
                pred = K_pred @ w.to(K_pred.device())
                mc_preds.append(pred)
            pred = torch.stack(mc_preds, dim=-1)
        else:
            pred = K_pred @ self.model.model.w
            pred = pred.unsqueeze(1)
        return pred

    def __repr__(self) -> str:
        return repr(self.model)
    
    def __str__(self) -> str:
        return str(self.model)
    