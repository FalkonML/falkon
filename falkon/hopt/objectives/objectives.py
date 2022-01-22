import abc
from typing import Optional

import torch

from falkon.hopt.objectives.transforms import PositiveTransform
from torch.distributions.transforms import identity_transform


class FakeTorchModelMixin(abc.ABC):
    def __init__(self):
        self._named_parameters = {}
        self._named_buffers = {}

    def register_parameter(self, name: str, param: torch.Tensor):
        if name in self._named_buffers:
            del self._named_buffers[name]
        self._named_parameters[name] = param

    def register_buffer(self, name: str, param: torch.Tensor):
        if name in self._named_parameters:
            del self._named_parameters[name]
        self._named_buffers[name] = param

    def parameters(self):
        return list(self._named_parameters.values())

    def named_parameters(self):
        return self._named_parameters

    def buffers(self):
        return list(self._named_buffers.values())

    def eval(self):
        pass


class HyperoptObjective2(torch.nn.Module):
    def __init__(self,
                 centers_init: torch.Tensor,
                 sigma_init: torch.Tensor,
                 penalty_init: torch.Tensor,
                 opt_centers: bool,
                 opt_sigma: bool,
                 opt_penalty: bool,
                 centers_transform: Optional[torch.distributions.Transform],
                 sigma_transform: Optional[torch.distributions.Transform],
                 pen_transform: Optional[torch.distributions.Transform],
                 ):
        super(HyperoptObjective2, self).__init__()

        self.centers_transform = centers_transform or identity_transform
        self.penalty_transform = pen_transform or PositiveTransform(1e-8)
        self.sigma_transform = sigma_transform or identity_transform  #PositiveTransform(1e-4)

        # Apply inverse transformations
        centers_init = self.centers_transform.inv(centers_init)
        penalty_init = self.penalty_transform.inv(penalty_init)
        sigma_init = self.sigma_transform.inv(sigma_init)

        if opt_centers:
            self.register_parameter("centers_", torch.nn.Parameter(centers_init))
        else:
            self.register_buffer("centers_", centers_init)
        if opt_sigma:
            self.register_parameter("sigma_", torch.nn.Parameter(sigma_init))
        else:
            self.register_buffer("sigma_", sigma_init)
        if opt_penalty:
            self.register_parameter("penalty_", torch.nn.Parameter(penalty_init))
        else:
            self.register_buffer("penalty_", penalty_init)

    @property
    def penalty(self):
        return self.penalty_transform(self.penalty_)

    @property
    def sigma(self):
        return self.sigma_transform(self.sigma_)

    @property
    def centers(self):
        return self.centers_

    @abc.abstractmethod
    def predict(self, X):
        pass


class HyperoptObjective(FakeTorchModelMixin, abc.ABC):
    losses_are_grads = False

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def hp_loss(self, X, Y):
        pass

    @property
    @abc.abstractmethod
    def loss_names(self):
        pass


class NKRRHyperoptObjective(HyperoptObjective, abc.ABC):
    def __init__(self, penalty, sigma, centers, cuda, verbose):
        super().__init__()
        if cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.centers_: Optional[torch.Tensor] = None
        self.sigma_: Optional[torch.Tensor] = None
        self.penalty_: Optional[torch.Tensor] = None

        self.penalty_transform = PositiveTransform(1e-9)
        self.penalty = penalty
        self.register_buffer("penalty", self.penalty_)

        self.sigma_transform = PositiveTransform(1e-4)
        self.sigma = sigma
        self.register_buffer("sigma", self.sigma_)

        self.centers = centers
        self.register_buffer("centers", self.centers_)

        self.verbose = verbose

    @property
    def penalty(self):
        return self.penalty_transform(self.penalty_)

    @property
    def sigma(self):
        return self.sigma_transform(self.sigma_)

    @property
    def centers(self):
        return self.centers_

    @centers.setter
    def centers(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.centers_ = value.clone().detach().to(device=self.device)

    @sigma.setter
    def sigma(self, value):
        # sigma cannot be a 0D tensor.
        if isinstance(value, float):
            value = torch.tensor([value], dtype=self.penalty_.dtype)
        elif isinstance(value, torch.Tensor) and value.dim() == 0:
            value = torch.tensor([value.item()], dtype=value.dtype)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.sigma_ = self.sigma_transform.inv(value.clone().detach().to(device=self.device))

    @penalty.setter
    def penalty(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.penalty_ = self.penalty_transform.inv(value.clone().detach().to(device=self.device))
