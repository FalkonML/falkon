from .compreg import CregNoTrace
from .gcv import NystromGCV
from .holdout import NystromHoldOut
from .loocv import NystromLOOCV
from .new_compreg import DeffPenFitTr
from .sgpr import SGPR

__all__ = (
    "CregNoTrace",
    "NystromGCV",
    "NystromHoldOut",
    "NystromLOOCV",
    "DeffPenFitTr",
    "SGPR",
)
