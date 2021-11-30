import abc
from typing import Optional

import torch

from falkon.hopt.objectives.transforms import PositiveTransform


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
        self.sigma_ = self.sigma_transform._inverse(value.clone().detach().to(device=self.device))

    @penalty.setter
    def penalty(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.penalty_ = self.penalty_transform._inverse(value.clone().detach().to(device=self.device))
