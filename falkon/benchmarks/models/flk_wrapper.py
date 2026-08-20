

import torch
from falkon import kernels


class FalkonWrapper:
    def __init__(self, base_model, kernel_type: str, kernel_sigma: float):
        self.base_model = base_model
        self.kernel_sigma = kernel_sigma
        self.kernel_type = kernel_type

    def get_median_sigma(self, data, num_samples=10000):
        sub_data = data[:num_samples]
        return torch.median(torch.pdist(sub_data))

    def get_kernel_params(self, X):
        sigma = self.kernel_sigma
        if sigma < 0:
            sigma = self.get_median_sigma(X).item()
            print(f"kernel sigma chosen with median heuristic: {sigma}")
        kernel_type = self.kernel_type
        if kernel_type.lower() == "gaussian":
            k = kernels.GaussianKernel(sigma)
        elif kernel_type.lower() == "laplacian":
            k = kernels.LaplacianKernel(sigma)
        elif kernel_type.lower() == "linear":
            k = kernels.LinearKernel(beta=1.0, gamma=sigma)
        else:
            raise ValueError(f"Kernel {kernel_type} not understood for algorithm Balkon")
        return k

    def fit(self, Xtr, Ytr, Xts, Yts):
        kernel = self.get_kernel_params(Xtr)
        self.base_model.kernel = kernel
        return self.base_model.fit(Xtr, Ytr, Xts, Yts)

    def predict(self, Xtst):
        return self.base_model.predict(Xtst)