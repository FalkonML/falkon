from abc import abstractmethod, ABC

import torch

__all__ = ("Loss", "LogisticLoss")

import falkon
from falkon.utils import CompOpt


class Loss(ABC):
    def __init__(self, name, kernel, opt=None, **kw):
        self.name = name
        self.kernel = kernel
        self.params = CompOpt(opt, **kw)

    def _update_opt(self, opt, **kw):
        new_opt = self.params.copy()
        if opt is not None:
            new_opt.update(opt)
        new_opt.update(kw)
        return new_opt

    @abstractmethod
    def __call__(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def df(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def ddf(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        pass

    def knmp_grad(self, X, Xc, Y, u, opt=None):
        """
        Calculate (1/n)* K(Xc, X) @ df(Y, K(X, Xc) @ u)

        And return also K(X, Xc) @ u
        """
        opt = self._update_opt(opt)
        func_val = self.kernel.mmv(X, Xc, u, opt=opt)
        grad = self.df(Y, func_val)
        out = self.kernel.mmv(Xc, X, grad, opt=opt)
        out.mul_(1/X.shape[0])
        return out, func_val

    def knmp_hess(self, X, Xc, Y, f, u, opt=None):
        """
        Calculate (1/n)* K(Xc, X) @ (ddf(Y, f) * K(X, Xc) @ u)

        Arguments:
        ----------
         - X : Tensor (N x D)
         - Xc : Tensor (M x D)
         - Y : Tensor (N x 1)
         - f : Tensor (N x 1)
         - u : Tensor (M x 1)
        """
        opt = self._update_opt(opt)
        inner = self.kernel.mmv(X, Xc, u, opt=opt)
        inner.mul_(self.ddf(Y, f))
        outer = self.kernel.mmv(Xc, X, inner, opt=opt)
        outer.mul_(1/X.shape[0])
        return outer

    def __str__(self):
        return self.name


class LogisticLoss(Loss):
    def __init__(self, kernel: falkon.kernels.Kernel, opt=None, **kw):
        super().__init__(name="LogisticLoss", kernel=kernel, opt=opt, **kw)

    def __call__(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(-y1 * y2))

    # noinspection PyUnresolvedReferences
    def df(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """
        -y1 / (1 + exp(y1 * y2))
        """
        out = -y1
        div = y1 * y2
        div.exp_().add_(1)
        out.div_(div)
        return out

    # noinspection PyUnresolvedReferences
    def ddf(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """
        """
        out = torch.pow(y1, 2)
        mul = y1 * y2
        mul.exp_()

        div = mul.reciprocal()
        div.add_(mul).add_(2)
        out.div_(div)
        return out

    def __repr__(self):
        return "LogisticLoss(kernel=%r)" % (self.kernel)
