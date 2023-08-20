from abc import ABC, abstractmethod


class Preconditioner(ABC):
    """Generic preconditioner class, used to accelerate solutions to linear systems.

    Given a system of equations :math:`H\\beta = Y`, where :math:`H` typically contains in some
    form our data matrix `X` and `Y` contains the targets. We can use matrix :math:`B` to
    create an equivalent linear system which will have lower condition number:

    .. math::

        BB^\\top H \\beta = Y

    where :math:`BB^\\top \\approx H^{-1}` in order to make the preconditioner effective, but not
    too expensive to compute. Then, in order to use the preconditioner in an algorithm based
    on matrix-vector products (such as conjugate gradient descent), we must be able to "apply" the
    matrix :math:`B` and its transpose :math:`B^\top` to any vector.

    For this reason, this class exposes abstract methods `apply` and `apply_t` which should
    be overridden in concrete preconditioner implementations

    See Also
    --------
    :class:`falkon.preconditioner.FalkonPreconditioner` :
        for an actual preconditioner implementation
    """
    @abstractmethod
    def apply(self, v):
        pass

    @abstractmethod
    def apply_t(self, v):
        pass
