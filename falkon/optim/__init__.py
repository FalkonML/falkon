from .conjgrad import ConjugateGradient, FalkonConjugateGradient
from .gd import GradientDescent, FalkonGradientDescent

__all__ = ('Optimizer', 'ConjugateGradient', 'FalkonConjugateGradient', 'GradientDescent',
           'FalkonGradientDescent')


class StopOptimizationException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


class Optimizer(object):
    """Base class for optimizers. This is an empty shell at the moment.
    """
    def __init__(self):
        pass
