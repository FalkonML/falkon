import torch

from falkon.options import FalkonOptions
from falkon.preconditioner.flk_pc import FalkonPreconditioner
from falkon.preconditioner.pc_utils import check_init
from falkon.sparse.sparse_tensor import SparseTensor
from falkon.utils import decide_cuda

from .preconditioner import Preconditioner


class BalkonPreconditioner(Preconditioner):
    def __init__(self, penalty: float, kernel, data_size: int, block_size: int, opt: FalkonOptions):
        super().__init__()
        self.params = opt
        self._use_cuda = decide_cuda(self.params) and not self.params.cpu_preconditioner

        self.X_nys: torch.Tensor | None = None

        self.base_prec = FalkonPreconditioner(penalty=penalty, kernel=kernel, opt=opt)
        self.block_size = block_size
        self.data_size = data_size

    def check_inputs(self, X: torch.Tensor | SparseTensor):
        if X.is_cuda and not self._use_cuda:
            raise RuntimeError("use_cuda is set to False, but data is CUDA tensor. Check your options.")

    def init(self, X: torch.Tensor):
        self.check_inputs(X)
        self.X_nys = X

    def to(self, device):
        if self.X_nys is not None:
            self.X_nys = self.X_nys.to(device)
        return self

    @check_init("X_nys")
    def apply(self, v: torch.Tensor) -> torch.Tensor:
        assert self.X_nys is not None
        M = self.X_nys.shape[0]
        num_blocks = M // self.block_size

        out = torch.empty_like(v)
        for i in range(num_blocks):
            i_start = i * self.block_size
            i_end = (i + 1) * self.block_size
            # TODO: Maybe we'd like an option to send smaller nystrom chunks to GPU.
            self.base_prec.init(self.X_nys[i_start:i_end])
            out[i_start:i_end] = self.base_prec.apply_t(self.base_prec.apply(v[i_start:i_end]))
        out = out.div_(num_blocks * self.data_size)
        return out

    def apply_t(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Balkon preconditioner does not support transpose application")

    def __str__(self):
        return (
            f"BalkonPreconditioner(block_size={self.block_size}, "
            f"_lambda={self.base_prec._lambda}, kernel={self.base_prec.kernel})"
        )
