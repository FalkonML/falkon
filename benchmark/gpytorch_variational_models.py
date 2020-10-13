import time

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy

__all__ = ("get_rbf_kernel", "RegressionVGP", "TwoClassVGP", "MultiClassVGP")


def _choose_var_dist(dist_str, num_points, batch_shape=1):
    if batch_shape == 1:
        batch_shape = torch.Size([])
    else:
        batch_shape = torch.Size([batch_shape])
    if dist_str == "diag":
        return gpytorch.variational.MeanFieldVariationalDistribution(
            num_points, batch_shape=batch_shape)
    elif dist_str == "full":
        return gpytorch.variational.CholeskyVariationalDistribution(
            num_points, batch_shape=batch_shape)
    elif dist_str == "delta":
        return gpytorch.variational.DeltaVariationalDistribution(
            num_points, batch_shape=batch_shape)
    elif dist_str == "natgrad":
        return gpytorch.variational.NaturalVariationalDistribution(
            num_points, batch_shape=batch_shape)
    else:
        raise KeyError(dist_str)


def _choose_var_strat(model, var_strat, var_dist, ind_pt, learn_ind=True, num_classes=None):
    if var_strat == "multi_task":
        try:
            num_classes = int(num_classes)
        except TypeError:
            raise RuntimeError("Multi-task variational strategy must specify integer num_classes")

        return gpytorch.variational.MultitaskVariationalStrategy(
            VariationalStrategy(model, ind_pt, var_dist, learn_inducing_locations=learn_ind),
            num_tasks=num_classes, task_dim=0,
        )
    else:
        return UnwhitenedVariationalStrategy(model, ind_pt, var_dist, learn_inducing_locations=learn_ind)


def get_rbf_kernel(ard=None, batch_shape=1):
    if batch_shape == 1:
        return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard))
    else:
        return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard, batch_shape=torch.Size([batch_shape])),
                    batch_shape=torch.Size([batch_shape]))


class BaseModel(ApproximateGP):
    def __init__(self, strategy, likelihood):
        super().__init__(strategy)
        self.strategy = strategy
        self.likelihood = likelihood

    def cuda(self):
        super().cuda()
        self.likelihood = self.likelihood.cuda()
        return self

    def parameters(self):
        return list(super().parameters()) + list(self.likelihood.parameters())

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



class GenericApproxGP(BaseModel):
    def __init__(self,
                 inducing_points,
                 mean_module,
                 covar_module,
                 var_strat: str,
                 learn_ind_pts: bool,
                 var_distrib: str,
                 likelihood):
        distribution = _choose_var_dist(
            var_distrib, inducing_points.size(-2), batch_shape=1).cuda()
        strategy = _choose_var_strat(
            self, var_strat, distribution, inducing_points, learn_ind=learn_ind_pts, num_classes=None).cuda()

        super().__init__(strategy, likelihood)

        self.mean_module = mean_module
        self.covar_module = covar_module

        if not strategy.variational_params_initialized.item():
            strategy._variational_distribution.initialize_variational_distribution(strategy.prior_distribution)
            strategy.variational_params_initialized.fill_(1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiTaskApproxGP(BaseModel):
    def __init__(self,
                 inducing_points,
                 mean_module,
                 covar_module,
                 var_strat: str,
                 learn_ind_pts: bool,
                 var_distrib: str,
                 likelihood,
                 batch_shape):
        distribution = _choose_var_dist(
            var_distrib, inducing_points.size(-2), batch_shape)
        strategy = _choose_var_strat(
            self, var_strat, distribution, inducing_points,
            learn_ind=learn_ind_pts, num_classes=batch_shape)

        super().__init__(strategy, likelihood)

        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPTrainer():
    def __init__(self,
                 model,
                 err_fn,
                 mb_size,
                 use_cuda,
                 mll,
                 num_epochs,
                 params,
                 natgrad_lr,
                 num_data,
                 likelihood,
                 lr=0.001):
        self.model = model
        self.mll = mll
        self.num_epochs = num_epochs
        self.lr = lr
        self.natgrad_lr = natgrad_lr

        self.err_fn = err_fn
        self.mb_size = mb_size

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()
        self.params = params
        # Count params
        num_params = [np.prod(p.data.shape) for p in params]
        print("Training with %d parameters" % (sum(num_params)))
        # Initialize optimizer with the parameters
        if self.natgrad_lr > 0:
            self.ng_optimizer = gpytorch.optim.NGD(self.model.variational_parameters(),
                    num_data=num_data, lr=self.natgrad_lr)
            params = set(list(self.model.hyperparameters()) + list(likelihood.parameters()))
            self.optimizer = torch.optim.Adam(list(params), lr=lr)
        else:
            self.ng_optimizer = None
            self.optimizer = torch.optim.Adam(self.params, lr=lr)

        self.error_every = 100

    def do_train(self, Xtr, Ytr, Xval, Yval):
        # Define dataset iterators
        train_dataset = torch.utils.data.TensorDataset(Xtr, Ytr)
        # Pinning memory of DataLoader results in slower training.
        if self.mb_size == 1:
            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=8)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.mb_size, shuffle=True, num_workers=8)

        # Start training
        t_elapsed = 0
        for epoch in range(self.num_epochs):
            t_start = time.time()
            self.model.train()
            for j, (x_batch, y_batch) in enumerate(train_loader):
                if self.use_cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                if self.ng_optimizer is not None:
                    self.ng_optimizer.zero_grad()
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -self.mll(output, y_batch)
                loss.backward()
                if self.ng_optimizer is not None:
                    self.ng_optimizer.step()
                self.optimizer.step()
                if j % self.error_every == 0:
                    t_elapsed += time.time() - t_start
                    err, err_name = self.err_fn(y_batch.cpu(), self.model.likelihood(output).mean.detach().cpu())
                    print('Epoch %d, iter %d/%d - Elapsed %.1fs - Loss: %.3f - %s: %.3f' %
                          (epoch + 1, j, len(train_loader), t_elapsed, loss.item(), err_name, err), flush=True)
                    t_start = time.time()
            t_elapsed += time.time() - t_start  # t_start will be reset at the start of the loop

            test_pred = self.predict(Xval)
            err, err_name = self.err_fn(Yval, test_pred)
            print("Epoch %d - elapsed %.2fs - validation %s: %.5f" % (epoch + 1, t_elapsed, err_name, err))
        print("Training took %.2fs" % (t_elapsed))

    def predict(self, X):
        test_dataset = torch.utils.data.TensorDataset(X)
        if self.mb_size > 1:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.mb_size, shuffle=False, num_workers=8)
        else:
            test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=8)

        self.model.eval()
        test_pred_means = []
        for x_batch in test_loader:
            x_batch = x_batch[0]
            if self.use_cuda:
                x_batch = x_batch.cuda()
            preds = self.model.likelihood(self.model(x_batch))
            test_pred_means.append(preds.mean.cpu().detach())

            del x_batch
        test_pred_means = torch.cat(test_pred_means)
        return test_pred_means


class RegressionVGP(GPTrainer):
    def __init__(self,
                 inducing_points,
                 kernel,
                 var_dist: str,
                 err_fn,
                 mb_size: int,
                 num_data: int,
                 num_epochs: int,
                 use_cuda: bool,
                 natgrad_lr: float,
                 lr: float=0.001,
                 learn_ind_pts: bool = False):
        self.var_dist = var_dist
        mean_module = gpytorch.means.ConstantMean()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if use_cuda:
            inducing_points = inducing_points.contiguous().cuda()
            mean_module = mean_module.cuda()
            kernel = kernel.cuda()

        model = GenericApproxGP(inducing_points,
                                mean_module=mean_module,
                                covar_module=kernel,
                                var_strat="var_strat",
                                var_distrib=var_dist,
                                likelihood=likelihood,
                                learn_ind_pts=learn_ind_pts,
                                )
        loss_fn = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
        params = model.parameters()
        if not learn_ind_pts:
            exclude = set(mean_module.parameters()) | set(kernel.parameters())
            print("Excluding parameters from mean and covariance models:", exclude)
            params = list(set(model.parameters()) - exclude)
        super().__init__(model, err_fn, mb_size, use_cuda, mll=loss_fn, num_epochs=num_epochs, lr=lr, params=params,
                         natgrad_lr=natgrad_lr, num_data=num_data, likelihood=likelihood)

    def do_train(self, Xtr, Ytr, Xts, Yts):
        super().do_train(Xtr, Ytr.reshape(-1), Xts, Yts.reshape(-1))

    def __str__(self):
        num_ind_pt = self.model.inducing_points.shape[0]
        ker = self.model.covar_module
        lengthscale = [p for name, p in dict(ker.named_parameters()).items() if 'raw_lengthscale' in name]
        num_ker_params = lengthscale[0].shape #dict(ker.named_parameters())['raw_lengthscale'].shape
        var_dist_params = self.model.variational_strategy._variational_distribution._parameters
        var_dist_num_params = sum([sum(p.shape) for p in var_dist_params.values()])
        return (f"RegressionVGP<num_inducing_points={num_ind_pt}, kernel={ker}, "
                f"kernel_params={num_ker_params}, likelihood={self.model.likelihood}, "
                f"variational_distribution={self.var_dist}, strategy={self.model.strategy}, "
                f"variational_params={var_dist_num_params}, "
                f"mini-batch={self.mb_size}, lr={self.lr}, natgrad_lr={self.natgrad_lr}>")


class TwoClassVGP(GPTrainer):
    def __init__(self,
                 inducing_points,
                 kernel,
                 var_dist: str,
                 err_fn,
                 mb_size: int,
                 num_data: int,
                 num_epochs: int,
                 use_cuda: bool,
                 natgrad_lr: float,
                 lr: float=0.001,
                 learn_ind_pts: bool = True):
        self.var_dist = var_dist
        if use_cuda:
            inducing_points = inducing_points.contiguous().cuda()

        mean_module = gpytorch.means.ConstantMean()
        # Only difference from regression is use of Bernoulli likelihood
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        model = GenericApproxGP(inducing_points,
                                mean_module=mean_module,
                                covar_module=kernel,
                                var_strat="var_strat",
                                var_distrib=var_dist,
                                likelihood=likelihood,
                                learn_ind_pts=learn_ind_pts,
                                )
        loss_fn = gpytorch.mlls.VariationalELBO(likelihood, model, num_data)
        params = model.parameters()
        if not learn_ind_pts:
            exclude = set(mean_module.parameters()) | set(kernel.parameters())
            print("Excluding parameters from mean and covariance models:", exclude)
            params = list(set(model.parameters()) - exclude)
        super().__init__(model, err_fn, mb_size, use_cuda, mll=loss_fn, num_epochs=num_epochs, lr=lr, params=params,
                         natgrad_lr=natgrad_lr, num_data=num_data, likelihood=likelihood)

    def do_train(self, Xtr, Ytr, Xts, Yts):
        Ytr = (Ytr + 1) / 2
        Yts = (Yts + 1) / 2
        super().do_train(Xtr, Ytr.reshape(-1), Xts, Yts.reshape(-1))

    def __str__(self):
        num_ind_pt = self.model.inducing_points.shape[0]
        ker = self.model.covar_module
        num_ker_params = 0
        for pn, pv in ker.named_parameters():
            if 'raw_lengthscale' in pn:
                num_ker_params = pv.shape
                continue
        var_dist_params = self.model.variational_strategy._variational_distribution._parameters
        var_dist_num_params = sum([sum(p.shape) for p in var_dist_params.values()])
        return (f"TwoClassVGP<num_inducing_points={num_ind_pt}, kernel={ker}, "
                f"kernel_params={num_ker_params}, likelihood={self.model.likelihood}, "
                f"variational_distribution={self.var_dist}, variational_params={var_dist_num_params}, "
                f"mini-batch={self.mb_size}, lr={self.lr}>")


class MultiClassVGP(GPTrainer):
    def __init__(self,
                 inducing_points,
                 kernel,
                 num_classes: int,
                 var_dist: str,
                 err_fn,
                 mb_size: int,
                 num_data: int,
                 num_epochs: int,
                 use_cuda: bool,
                 natgrad_lr: float,
                 lr: float = 0.001,
                 learn_ind_pts: bool = True):
        #if mb_size != 1:
        #    raise ValueError("MultiTask VGP must be run with batch size of 1.")
        self.num_classes = num_classes
        self.var_dist = var_dist
        if use_cuda:
            inducing_points = inducing_points.contiguous().cuda()

        mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes]))
        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(#num_tasks=num_classes)
             num_classes=num_classes,
             mixing_weights=False)
        model = MultiTaskApproxGP(inducing_points,
                                  mean_module=mean_module,
                                  covar_module=kernel,
                                  var_strat="multi_task",
                                  var_distrib=var_dist,
                                  likelihood=likelihood,
                                  batch_shape=num_classes,
                                  learn_ind_pts=learn_ind_pts,)
        loss_fn = gpytorch.mlls.VariationalELBO(likelihood, model, num_data)
        params = model.parameters()
        if not learn_ind_pts:
            exclude = set(mean_module.parameters()) + set(kernel.parameters())
            params = list(set(model.parameters()) - exclude)
        super().__init__(model, err_fn, mb_size, use_cuda, mll=loss_fn, num_epochs=num_epochs, lr=lr, params=params,
                         natgrad_lr=natgrad_lr, num_data=num_data, likelihood=likelihood)

    def do_train(self, Xtr, Ytr, Xts, Yts):
        super().do_train(Xtr, Ytr, Xts, Yts)

    def __str__(self):
        num_ind_pt = self.model.inducing_points.shape[0]
        ker = self.model.covar_module
        num_ker_params = ker._parameters['raw_lengthscale'].shape
        var_dist_params = self.model.variational_strategy.base_variational_strategy._variational_distribution._parameters
        var_dist_num_params = sum([sum(p.shape) for p in var_dist_params.values()])
        return (f"MultiClassVGP<num_inducing_points={num_ind_pt}, kernel={ker}, num_classes={self.num_classes}, "
                f"kernel_params={num_ker_params}, likelihood={self.model.likelihood}, "
                f"variational_distribution={self.var_dist}, variational_params={var_dist_num_params}, "
                f"mini-batch={self.mb_size}, lr={self.lr}>")
