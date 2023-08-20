import abc
from typing import Optional

import torch

import falkon.kernels
from falkon.hopt.objectives.transforms import PositiveTransform
from torch.distributions.transforms import identity_transform


class HyperoptObjective(torch.nn.Module):
    def __init__(self,
                 kernel: falkon.kernels.DiffKernel,
                 centers_init: torch.Tensor,
                 penalty_init: torch.Tensor,
                 opt_centers: bool,
                 opt_penalty: bool,
                 centers_transform: Optional[torch.distributions.Transform],
                 pen_transform: Optional[torch.distributions.Transform],
                 ):
        """

        Parameters
        ----------
        kernel
        centers_init
        penalty_init
        opt_centers
        opt_penalty
        centers_transform
        pen_transform
        """
        super().__init__()

        if not isinstance(kernel, falkon.kernels.DiffKernel):
            raise TypeError("Kernel must inherit from `DiffKernel` for hyperparameter optimization.")
        self.kernel = kernel

        self.centers_transform = centers_transform or identity_transform
        self.penalty_transform = pen_transform or PositiveTransform(1e-8)

        # Apply inverse transformations
        centers_init = self.centers_transform.inv(centers_init)
        penalty_init = self.penalty_transform.inv(penalty_init)

        if opt_centers:
            self.register_parameter("centers_", torch.nn.Parameter(centers_init))
        else:
            self.register_buffer("centers_", centers_init)
        if opt_penalty:
            self.register_parameter("penalty_", torch.nn.Parameter(penalty_init))
        else:
            self.register_buffer("penalty_", penalty_init)

    @property
    def penalty(self):
        return self.penalty_transform(self.penalty_)

    @property
    def centers(self):
        return self.centers_

    @abc.abstractmethod
    def predict(self, X):
        pass
