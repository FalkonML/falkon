import dataclasses
from abc import abstractmethod, ABC
from typing import Optional, Tuple

import torch

import falkon
from falkon.options import FalkonOptions

__all__ = ("Loss", "LogisticLoss", "WeightedCrossEntropyLoss")


class Loss(ABC):
    r"""Abstract generalized self-concordant loss function class.

    Such loss functions must be three times differentiable; but for the logistic Falkon algorithm
    only the first two derivatives are used.
    Subclasses must implement the :meth:`__call__` method which calculates the loss function
    given two input vectors (the inputs could also be matrices e.g. for the softmax loss),
    the :meth:`df` method which calculates the first derivative of the function and :meth:`ddf`
    which calculates the second derivative.

    Additionally, this class provides two methods (:meth:`knmp_grad` and :meth:`knmp_hess`) which
    calculate kernel-vector products using the loss derivatives for vectors. These functions are
    specific to the logistic Falkon algorithm.

    Parameters
    -----------
    name
        A descriptive name for the loss function (e.g. "logistic", "softmax")
    kernel
        The kernel function used for training a LogFalkon model
    opt
        Falkon options container. Will be passed to the kernel when computing kernel-vector
        products.

    See Also
    --------
    :class:`LogisticLoss` : a concrete implementation of this class for the logistic loss.
    :class:`falkon.models.LogisticFalkon` : the logistic Falkon model which uses GSC losses.
    """

    def __init__(self,
                 name: str,
                 kernel: falkon.kernels.kernel.Kernel,
                 opt: Optional[FalkonOptions] = None):
        self.name = name
        self.kernel = kernel
        self.params = opt or FalkonOptions()

    def _update_opt(self, opt: Optional[FalkonOptions]):
        new_opt = self.params
        if opt is not None:
            new_opt = dataclasses.replace(self.params, **dataclasses.asdict(opt))
        return new_opt

    @abstractmethod
    def __call__(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Abstract method. Should return the loss for predicting `y2` with true labels `y1`.

        Parameters
        ----------
        y1 : torch.Tensor
            One of the two inputs to the loss. This should be interpreted as the `true` labels.
        y2 : torch.Tensor
            The other loss input. Should be interpreted as the predicted labels.

        Returns
        -------
        torch.Tensor
            The loss calculated for the two inputs.
        """
        pass

    @abstractmethod
    def df(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Abstract method. Should return the derivative of the loss wrt `y2`.

        Parameters
        ----------
        y1 : torch.Tensor
            One of the two inputs to the loss. This should be interpreted as the `true` labels.
        y2 : torch.Tensor
            The other loss input. Should be interpreted as the predicted labels. The derivative
            should be computed with respect to this tensor.

        Returns
        -------
        torch.Tensor
            The derivative of the loss with respect to `y2`. It will be a tensor of the same shape
            as the two inputs.
        """
        pass

    @abstractmethod
    def ddf(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Abstract method. Should return the second derivative of the loss wrt `y2`.

        Parameters
        ----------
        y1 : torch.Tensor
            One of the two inputs to the loss. This should be interpreted as the `true` labels.
        y2 : torch.Tensor
            The other loss input. Should be interpreted as the predicted labels. The derivative
            should be computed with respect to this tensor.

        Returns
        -------
        torch.Tensor
            The second derivative of the loss with respect to `y2`. It will be a tensor of the
            same shape as the two inputs.
        """
        pass

    def knmp_grad(self,
                  X: torch.Tensor,
                  Xc: torch.Tensor,
                  Y: torch.Tensor,
                  u: torch.Tensor,
                  opt: Optional[FalkonOptions] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computes a kernel vector product where the vector is the first derivative of this loss

        Given kernel function :math:`K`, the loss represented by this class :math:`\mathcal{l}`,
        number of samples :math:`n`, this function follows equation

        .. math::

            \dfrac{1}{n} K(X_c, X) @ (\mathcal{l}'(Y, K(X, X_c) @ u))

        Parameters
        ----------
        X : torch.Tensor
            Data matrix of shape (n x d) with `n` samples in `d` dimensions.
        Xc : torch.Tensor
            Center matrix of shape (m x d) with `m` centers in `d` dimensions.
        Y : torch.Tensor
            Label matrix of shape (n x t) with `n` samples. Depending on the loss, the labels may or may not
            have more than one dimension.
        u : torch.Tensor
            A vector (or matrix if the labels are multi-dimensional) of weights of shape (m x t).
            The product `K(X, Xc) @ u`, where `K` is the kernel associated to this loss, should
            produce label predictions.
        opt : FalkonOptions or None
            Options to be passed to the mmv function for the kernel associated to this loss.
            Options passed as an argument take precedence over the options used to build this
            class instance.

        Returns
        -------
        grad_mul : torch.Tensor
            A tensor of shape (m x 1) coming from the multiplication of the kernel matrix
            `K(Xc, X)` and the loss calculated on predictions with weights `u`.
            The formula followed is: `(1/n) * K(Xc, X) @ df(Y, K(X, Xc) @ u)`.
        func_val : torch.Tensor
            A tensor of shape (n x t) of predictions obtained with weights `u`.
        """
        opt = self._update_opt(opt)
        func_val = self.kernel.mmv(X, Xc, u, opt=opt)
        grad = self.df(Y, func_val)
        out = self.kernel.mmv(Xc, X, grad, opt=opt)
        out.mul_(1 / X.shape[0])
        return out, func_val

    def knmp_hess(self,
                  X: torch.Tensor,
                  Xc: torch.Tensor,
                  Y: torch.Tensor,
                  f: torch.Tensor,
                  u: torch.Tensor,
                  opt: Optional[FalkonOptions] = None) -> torch.Tensor:
        r"""Compute a kernel-vector product with a rescaling with the second derivative

        Given kernel function :math:`K`, the loss represented by this class :math:`\mathcal{l}`,
        number of samples :math:`n`, this function follows equation

        .. math::

            \dfrac{1}{n} K(X_c, X) @ (\mathcal{l}''(Y, f) * K(X, X_c) @ u)

        Parameters
        ----------
        X : torch.Tensor
            Data matrix of shape (n x d) with `n` samples in `d` dimensions.
        Xc : torch.Tensor
            Center matrix of shape (m x d) with `m` centers in `d` dimensions.
        Y : torch.Tensor
            Label matrix of shape (n x t) with `n` samples. Depending on the loss, the labels may
            or may not have more than one dimension.
        f : torch.Tensor
            Tensor of shape (n x t) of predictions. Typically this will be the second output of
            the :meth:`knmp_grad` method.
        u : torch.Tensor
            A vector (or matrix if the labels are multi-dimensional) of weights of shape (m x t).
            The product `K(X, Xc) @ u`, where `K` is the kernel associated to this loss, should
            produce label predictions.
        opt : FalkonOptions or None
            Options to be passed to the mmv function for the kernel associated to this loss.
            Options passed as an argument take precedence over the options used to build this
            class instance.

        Returns
        -------
        A tensor of shape (m x t), the output of the computation.
        """
        opt = self._update_opt(opt)
        inner = self.kernel.mmv(X, Xc, u, opt=opt)
        inner.mul_(self.ddf(Y, f))
        outer = self.kernel.mmv(Xc, X, inner, opt=opt)
        outer.mul_(1 / X.shape[0])
        return outer

    def __str__(self):
        return self.name


class LogisticLoss(Loss):
    """Wrapper for the logistic loss, to be used in conjunction with the :class:`~falkon.models.LogisticFalkon` estimator.

    Parameters
    -----------
    kernel : falkon.kernels.kernel.Kernel
        The kernel function used for training a :class:`~falkon.models.LogisticFalkon` model
    opt : FalkonOptions
        Falkon options container. Will be passed to the kernel when computing kernel-vector
        products.

    Examples
    --------

    >>> k = falkon.kernels.GaussianKernel(3)
    >>> log_loss = LogisticLoss(k)
    >>> estimator = falkon.LogisticFalkon(k, [1e-4, 1e-4, 1e-4], [3, 3, 3], loss=log_loss, M=100)

    """

    def __init__(self, kernel: falkon.kernels.kernel.Kernel, opt: Optional[FalkonOptions] = None):
        super().__init__(name="LogisticLoss", kernel=kernel, opt=opt)

    def __call__(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        r"""Compute the logistic loss between two 1-dimensional tensors

        The formula used is :math:`\log(1 + \exp(-y_1 * y_2))`

        Parameters
        ----------
        y1
            The first input tensor. Must be 1D
        y2
            The second input tensor. Must be 1D

        Returns
        -------
        loss
            The logistic loss between the two input vectors.
        """
        return torch.log(1 + torch.exp(-y1 * y2))

    def df(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        r"""Compute the derivative of the logistic loss with respect to `y2`

        The formula used is

        .. math::

            -y_1 / (1 + \exp{y_1 * y_2})

        Parameters
        ----------
        y1
            The first input tensor. Must be 1D
        y2
            The second input tensor. Must be 1D

        Returns
        -------
        d_loss
            The derivative of the logistic loss, calculated between the two input vectors.
        """
        out = -y1
        div = y1 * y2
        div.exp_().add_(1)
        out.div_(div)
        return out

    def ddf(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        r"""Compute the second derivative of the logistic loss with respect to `y2`

        The formula used is

        .. math::

            y_1^2 \dfrac{1}{1 + \exp{-y_1 * y_2}} \dfrac{1}{1 + \exp{y_1 * y_2}}


        Parameters
        ----------
        y1
            The first input tensor. Must be 1D
        y2
            The second input tensor. Must be 1D

        Returns
        -------
        dd_loss
            The second derivative of the logistic loss, calculated between the two input vectors.
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
    
class WeightedCrossEntropyLoss(Loss):

    def __init__(self, kernel: falkon.kernels.Kernel, weight: float, opt: FalkonOptions = FalkonOptions()):
        super().__init__(name="WeightedCrossEntropy", kernel=kernel, opt=opt)
        self.weight = weight

    def __call__(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        
        class1=true*torch.log(1 + torch.exp(-pred))
        class0 =self.weight*(1 - true)*torch.log(1 + torch.exp(pred))        
        
        return (class1+class0)

    def df(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:

        num = (true * self.weight - self.weight)*torch.exp(pred) + true
        
        den = torch.exp(pred) +1 

        return -num/den

    def ddf(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        num = ((self.weight - 1)*true - self.weight)*torch.exp(pred)
        den = torch.pow((torch.exp(pred) + 1),2)
        return -num/den

    def __repr__(self):
        return "WeightedCrossEntropy(kernel=%r)" % (self.kernel)
