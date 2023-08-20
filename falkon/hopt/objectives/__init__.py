from .exact_objectives.compreg import CompReg
from .exact_objectives.gcv import GCV
from .exact_objectives.holdout import HoldOut
from .exact_objectives.loocv import LOOCV
from .exact_objectives.new_compreg import NystromCompReg
from .exact_objectives.sgpr import SGPR
from .stoch_objectives.stoch_new_compreg import StochasticNystromCompReg

__all__ = (
    "CompReg",
    "NystromCompReg",
    "HoldOut",
    "SGPR",
    "GCV",
    "LOOCV",
    "StochasticNystromCompReg",
)
