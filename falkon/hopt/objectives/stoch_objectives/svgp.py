import warnings

import torch
import gpytorch

from falkon.benchmarks.models.gpytorch_variational_models import GenericApproxGP

from falkon.hopt.objectives.objectives import NKRRHyperoptObjective
from falkon.hopt.utils import get_scalar


class SVGP(NKRRHyperoptObjective):
    def __init__(
            self,
            sigma_init,
            penalty_init,
            centers_init,
            opt_centers,
            opt_sigma,
            opt_penalty,
            cuda: bool,
            num_data: int,
            multi_class: int,
    ):
        """
        super().__init__(
            penalty=penalty_init,
            sigma=sigma_init,
            centers=centers_init,
            cuda=cuda,
            verbose=True,
        )
        """
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if not opt_sigma or not opt_penalty:
            raise ValueError("Sigma, Penalty must be optimized...")
        self.opt_sigma, self.opt_centers, self.opt_penalty = opt_sigma, opt_centers, opt_penalty
        self.variational_distribution = "diag"
        self.num_data = num_data

        if multi_class > 1:
            self.kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=sigma_init.shape[0], batch_shape=torch.Size([1]))
            mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=multi_class, has_task_noise=False)
            var_strat = "multi_task"
        else:
            if len(sigma_init) == 1:
                self.kernel = gpytorch.kernels.RBFKernel(ard_num_dims=None)
            else:
                self.kernel = gpytorch.kernels.RBFKernel(ard_num_dims=sigma_init.shape[0])
            mean_module = gpytorch.means.ConstantMean()
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            var_strat = "var_strat"
        self.penalty = penalty_init

        self.kernel.lengthscale = sigma_init
        if cuda:
            mean_module = mean_module.cuda()
            self.kernel = self.kernel.cuda()
        if centers_init.dtype == torch.float64:
            mean_module = mean_module.double()
            self.kernel = self.kernel.double()
        self.model = GenericApproxGP(centers_init.to(device=self.device),
                                     mean_module=mean_module,
                                     covar_module=self.kernel,
                                     var_strat=var_strat,
                                     var_distrib=self.variational_distribution,
                                     likelihood=self.likelihood,
                                     learn_ind_pts=self.opt_centers,
                                     num_classes=multi_class,
                                     cuda=cuda
                                     )
        self.loss_fn = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.num_data)
        if cuda:
            self.model = self.model.cuda()
        if centers_init.dtype == torch.float64:
            self.model = self.model.double()

    @property
    def penalty(self):
        try:
            return self.likelihood.noise / self.num_data
        except AttributeError:
            return 0.0

    @property
    def sigma(self):
        return self.kernel.lengthscale

    @property
    def centers(self):
        try:
            return self.model.strategy.inducing_points
        except AttributeError:
            return self.model.strategy.base_variational_strategy.inducing_points

    @centers.setter
    def centers(self, value):
        raise NotImplementedError("No setter implemented for centers")

    @sigma.setter
    def sigma(self, value):
        self.kernel.lengthscale = value
        #raise NotImplementedError("No setter implemented for sigma")

    @penalty.setter
    def penalty(self, value):
        try:
            #print("Setting likelihood noise to ", value * self.num_data)
            self.likelihood.noise = value * self.num_data
            #print("Likelihood noise is %f, raw %f" % (self.likelihood.noise, self.likelihood.raw_noise))
        except RuntimeError:
            warnings.warn(f"Setting lambda to {value * self.num_data} failed. Setting to 1e-4")
            self.likelihood.noise = 1e-4
        #raise NotImplementedError("No setter implemented for penalty")

    def hp_loss(self, X, Y):
        with gpytorch.settings.fast_computations(False, False, False):
            output = self.model(X)
            if Y.shape[1] == 1:
                Y = Y.reshape(-1)
            loss = -self.loss_fn(output, Y)
            return (loss,)

    def predict(self, X):
        with gpytorch.settings.fast_computations(False, False, False):
            preds = self.model.likelihood(self.model(X))
            return preds.mean.T

    @property
    def loss_names(self):
        return ("mll", )

    def named_parameters(self):
        return dict(self.model.named_parameters())

    def parameters(self):
        return list(self.named_parameters().values())

    def __repr__(self):
        return f"SVGP(sigma={get_scalar(self.sigma)}, penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}," \
               f"opt_centers={self.opt_centers}, opt_sigma={self.opt_sigma}, opt_penalty={self.opt_penalty}," \
               f"model={self.model})"
