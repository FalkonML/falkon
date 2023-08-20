import os

from torch.utils.tensorboard import SummaryWriter

__all__ = ("get_writer",)

LOG_DIR = "./logs/tensorboard"
_writer = None


def get_writer(name=None):
    global _writer
    if _writer is not None:
        return _writer

    log_dir = LOG_DIR
    if name is not None:
        log_dir = os.path.join(log_dir, name)

    _writer = SummaryWriter(log_dir=log_dir, max_queue=5, flush_secs=30)
    return _writer
