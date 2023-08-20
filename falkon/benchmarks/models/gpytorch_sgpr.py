import time

import gpytorch
import torch


class SGPRBaseModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()

        base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=None))
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel, inducing_points=inducing_points, likelihood=likelihood
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.likelihood = self.likelihood.cuda(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        return list(super().parameters(recurse)) + list(self.likelihood.parameters(recurse))

    def train(self, arg=True):
        super().train(arg)
        self.likelihood.train(arg)

    def eval(self):
        self.train(False)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param
        return None


class GpytorchSGPR:
    def __init__(
        self, inducing_points, err_fn, num_epochs: int, use_cuda: bool, lr: float = 0.001, learn_ind_pts: bool = False
    ):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.use_cuda = use_cuda
        self.inducing_points = inducing_points
        self.learn_ind_pts = learn_ind_pts

        self.lr = lr
        self.num_epochs = num_epochs
        self.err_fn = err_fn

        if use_cuda:
            self.inducing_points = self.inducing_points.contiguous().cuda()
            self.likelihood = self.likelihood.cuda()

    def do_train(self, Xtr, Ytr, Xts, Yts):
        Ytr = Ytr.reshape(-1)
        Yts = Yts.reshape(-1)

        self.model = SGPRBaseModel(Xtr, Ytr, self.likelihood, self.inducing_points)
        if self.use_cuda:
            self.model = self.model.cuda()

        # Loss function for the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Parameters of the model which will be trained
        params = self.model.parameters()
        if not self.learn_ind_pts:
            exclude = {self.model.inducing_points}
            print("Excluding inducing points from the model:", exclude)
            params = list(set(self.model.parameters()) - exclude)

        # Define optimizer
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Start training
        t_elapsed = 0
        if self.use_cuda:
            Xtr = Xtr.cuda()
            Ytr = Ytr.cuda()
        for epoch in range(self.num_epochs):
            # Train
            t_start = time.time()
            self.model.train()
            optimizer.zero_grad()
            output = self.model(Xtr)
            loss = -mll(output, Ytr)
            loss.backward()
            optimizer.step()
            t_elapsed += time.time() - t_start
            # Evaluate
            torch.cuda.empty_cache()
            err, err_name = self.err_fn(Yts, self.predict(Xts))
            print(
                "Epoch %d - Elapsed %.1fs - Train loss: %.3f - Test %s: %.3f"
                % (epoch + 1, t_elapsed, loss.item(), err_name, err),
                flush=True,
            )
            torch.cuda.empty_cache()
        print("Training took %.2fs" % (t_elapsed))

    def predict(self, X):
        self.model.eval()
        if self.use_cuda:
            X = X.cuda()
        preds = self.model.likelihood(self.model(X)).mean.cpu().detach()
        return preds

    def __str__(self):
        num_ind_pt = self.model.inducing_points.shape[0]
        ker = self.model.covar_module
        lengthscale = [p for name, p in dict(ker.named_parameters(recurse=True)).items() if "raw_lengthscale" in name]
        num_ker_params = lengthscale[0].shape
        return (
            f"RegressionVGP<num_inducing_points={num_ind_pt}, "
            f"learned_ind_pts={self.learn_ind_pts}, kernel={ker}, "
            f"kernel_params={num_ker_params}, likelihood={self.model.likelihood}, "
            f"lr={self.lr}>"
        )
