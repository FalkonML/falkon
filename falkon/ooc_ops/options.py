from dataclasses import dataclass
from typing import Union


@dataclass(frozen=False)
class LauumOptions:
    lauum_par_blk_multiplier: int = 8


@dataclass(frozen=False)
class CholeskyOptions:
    chol_block_size: int = 256
    chol_tile_size: Union[int, str] = 'auto'
    chol_force_in_core: bool = False
    chol_force_ooc: bool = False
    chol_force_parallel: bool = False
    chol_par_blk_multiplier: int = 2
