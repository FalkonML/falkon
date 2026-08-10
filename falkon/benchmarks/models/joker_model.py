import time

import numpy as np
import torch
from joker import InexactJoker, Joker
from kernels.kernel import make_kernel


class JokerWrapper:
    def __init__(
        self,
        inexact_type,
        dtype,
        crit,
        device,
        kernel_type: str,
        kernel_sigma: float,
        nrff: int,
        nfastfood: int,
        block_size: int,
        incore: bool,
        data_block_size: int,
        opt_name: str,
        num_iter: int,
        num_iter_subprob: int,
        max_region_size: float,
        region_shrink_freq: int,
        region_shrink_rate: float,
    ):
        self.inexact_type = inexact_type
        self.dtype = dtype
        self.kernel_type = kernel_type
        self.kernel_sigma = kernel_sigma
        self.crit = crit
        self.device = device
        self.nrff = nrff
        self.nfastfood = nfastfood
        self.block_size = block_size
        self.incore = incore
        self.data_block_size = data_block_size
        self.opt_name = opt_name
        self.num_iter = num_iter
        self.num_iter_subprob = num_iter_subprob
        self.max_region_size = max_region_size
        self.region_shrink_freq = region_shrink_freq
        self.region_shrink_rate = region_shrink_rate
        self.fit_times_ = []
        self.model = None

    def get_median_sigma_square(self, data, num_samples=10000):
        sub_data = data[:num_samples]
        return torch.median(torch.pdist(sub_data) ** 2)

    def get_kernel(self, Xtr):
        if self.kernel_sigma < 0:
            sigma_square = self.get_median_sigma_square(Xtr, num_samples=10_000)
            print(f"Using sigma = {float(np.sqrt(sigma_square)):.4f} from the median trick.")
        else:
            sigma_square = self.kernel_sigma ** 2
        if self.kernel_type == "rbf":
            gamma = 0.5 / sigma_square
        elif self.kernel_type == "lap":
            gamma = 1 / float(np.sqrt(sigma_square))
        else:
            raise RuntimeError(self.kernel_type)
        return make_kernel(self.kernel_type, gamma=gamma)

    def init_model(self, Xtr, Ytr):
        kernel_func = self.get_kernel(Xtr)
        if self.inexact_type == 'rff':
            model = InexactJoker(
                Xtr,
                Ytr,
                dtype=self.dtype,
                kernel=kernel_func,
                criterion=self.crit,
                device=self.device,
                n_features=self.nrff,
                inexact_type=self.inexact_type,
                opt_blksz=self.block_size, #cfg["blksz"],
                incore=self.incore,#cfg["incore"],
                data_blksz=self.data_block_size,
                optim=self.opt_name,
            )
        elif self.inexact_type == 'fastfood':
            model = InexactJoker(
                Xtr,
                Ytr,
                dtype=self.dtype,
                kernel=kernel_func,
                criterion=self.crit,
                device=self.device,
                n_features=self.nfastfood,
                inexact_type=self.inexact_type,
                opt_blksz=self.block_size, #cfg["blksz"],
                incore=self.incore,#cfg["incore"],
                data_blksz=self.data_block_size,
                optim=self.opt_name,
            )
        else:
            model = Joker(
                Xtr,
                Ytr,
                dtype=self.dtype,
                kernel=kernel_func,
                criterion=self.crit,
                device=self.device,
                opt_blksz=self.block_size, #cfg["blksz"],
                incore=self.incore,#cfg["incore"],
                data_blksz=self.data_block_size,
                optim=self.opt_name,
            )
        return model

    def fit(self, Xtr, Ytr, Xts, Yts):
        self.fit_times_ = []
        t_start = time.time()
        self.model = self.init_model(Xtr, Ytr)
        self.model.fit(
            max_iter=self.num_iter,
            max_iter_subprob=self.num_iter_subprob, #cfg["max_iter_subprob"],
            max_region_size=self.max_region_size, #cfg["max_trust_region_size"],
            region_shrink_freq=self.region_shrink_freq,
            verbose_freq=self.num_iter + 1,
            region_shrink_rate=self.region_shrink_rate, #cfg["region_shrink_rate"],
            blk_strategy='random', #cfg["blk_strategy"],
            val_x=Xts,
            val_y=Yts,
            verbose_primal_dual=False
        )
        self.fit_times_.append(time.time() - t_start)

    def predict(self, Xtst):
        if self.model is None:
            raise RuntimeError("predict called before fit")
        return self.model.predict(Xtst)


