class StopOptimizationException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


class Optimizer(object):
    """Base class for optimizers. This is an empty shell at the moment.
    """
    def __init__(self):
        pass


from .conjgrad import ConjugateGradient, FalkonConjugateGradient  # noqa E402
from .gd import GradientDescent, FalkonGradientDescent  # noqa E402
from .sgd import FalkonSGD  # noqa E402

__all__ = ('Optimizer', 'StopOptimizationException',
           'ConjugateGradient', 'FalkonConjugateGradient',
           'GradientDescent', 'FalkonGradientDescent',
           'FalkonSGD'
           )
