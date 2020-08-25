try:
    from .ooc_lauum import gpu_lauum
    from .ooc_potrf import gpu_cholesky
except OSError:
    # No GPU
    gpu_lauum = None
    gpu_cholesky = None

__all__ = ("gpu_lauum", "gpu_cholesky")