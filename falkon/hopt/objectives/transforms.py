import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F


class TransformedParameter():
    def __init__(self, value, transform):
        self.transform = transform
        self.value = self.transform.forward(value)

    def __get__(self):
        return self.value

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__repr__()


class ExpTransform(torch.distributions.transforms.Transform):
    _cache_size = 0
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, ExpTransform)

    def _call(self, x):
        return torch.exp(x)

    def _inverse(self, y):
        return torch.log(y)


class PositiveTransform(torch.distributions.transforms.Transform):
    _cache_size = 0
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self, lower_bound=0.0):
        super().__init__()
        self.lower_bound = lower_bound

    def __eq__(self, other):
        if not isinstance(other, PositiveTransform):
            return False
        return other.lower_bound == self.lower_bound

    def _call(self, x):
        # softplus and then shift
        y = F.softplus(x)
        y = y + self.lower_bound
        return y

    def _inverse(self, y):
        # https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/math/generic.py#L456-L507
        x = y - self.lower_bound

        threshold = torch.log(torch.tensor(torch.finfo(y.dtype).eps, dtype=y.dtype)) + torch.tensor(2., dtype=y.dtype)
        is_too_small = x < torch.exp(threshold)
        is_too_large = x > -threshold
        too_small_val = torch.log(x)
        too_large_val = x

        x = torch.where(is_too_small | is_too_large, torch.tensor(1.0, dtype=y.dtype, device=y.device), x)
        x = x + torch.log(-torch.expm1(-x))
        return torch.where(is_too_small,
                           too_small_val,
                           torch.where(is_too_large, too_large_val, x))
