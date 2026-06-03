
import time

import numpy as np
import torch

import eigenpro.models.sharded_kernel_machine as skm # pyright: ignore[reportMissingImports]
import eigenpro.solver as solver # pyright: ignore[reportMissingImports]


class EigenProWrapper():
    def __init__(self, device, dtype: torch.dtype, kernel_fn, num_centers, num_pc_centers, num_eigenvalues, num_epochs):
        self.device = device
        self.dtype = dtype
        self.kernel_fn = kernel_fn
        self.num_centers = num_centers
        self.num_pc_centers = num_pc_centers
        self.num_eigenvalues = num_eigenvalues
        self.num_epochs = num_epochs

        self.epoch_times = []
        self.model = None
        self.batch_size = 8192

    def inter_epoch_cback(self, Xts, Yts, err_fns):
        def fn(model):
            start_time = self.epoch_times[-1]
            elapsed_time = time.time() - start_time
            self.epoch_times[-1] = elapsed_time
            epoch = len(self.epoch_times)
            self.model = model
            print("Running test-set predictions...", flush=True)
            pred_start_time = time.time()
            preds = self.predict(Xts)
            pred_elapsed = time.time() - pred_start_time
            print(f"EigenPro4 epoch {epoch}:")
            print(f"\telapsed: {sum(self.epoch_times):.2f}s - predictions in {pred_elapsed:.2f}s", flush=True)
            for err_fn in err_fns:
                test_err, test_err_name = err_fn(Yts, preds)
                print(f"\ttest {test_err_name}: {test_err:9.6f}", flush=True)
            print()
            self.epoch_times.append(time.time())
        return fn

    def fit(self, Xtr, Ytr, Xts, Yts, err_fns):
        centers_set_indices = np.random.choice(
            Xtr.shape[0], self.num_centers, replace=False
        )
        Z = Xtr[centers_set_indices, :]
        kernel_model = skm.create_sharded_kernel_machine(
            Z, Ytr.shape[-1], self.kernel_fn, self.device, dtype=self.dtype, tmp_centers_coeff=2
        )
        self.epoch_times.append(time.time())
        self.model = solver.fit(
            kernel_model, Xtr, Ytr, Xts, Yts, self.device,
            dtype=self.dtype, kernel=self.kernel_fn, n_data_pcd_nyst_samples=self.num_pc_centers,
            n_model_pcd_nyst_samples=self.num_pc_centers, n_data_pcd_eigenvals=self.num_eigenvalues,
            n_model_pcd_eigenvals=self.num_eigenvalues, epochs=self.num_epochs,
            accumulated_gradients=True, callback=self.inter_epoch_cback(Xts, Yts, err_fns)
        )
        return self

    def predict(self, data, batch_size=None):
        if self.model is None:
            raise ValueError("predict called without previous fit. Call fit.")
        if batch_size is None:
            batch_size = self.batch_size
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        outputs = []
        for i in range(0, data.shape[0], batch_size):
            batch = data[i: i + batch_size].to(device=device)
            outputs.append(self.model(batch).cpu())
        return torch.cat(outputs, 0)
        

