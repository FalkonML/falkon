import warnings

from falkon.options import BaseOptions, KeopsOptions


def decide_cuda(opt: BaseOptions = BaseOptions()):
    if opt.use_cpu:
        return False

    def get_error_str(name, err):
        return ("Failed to initialize %s library; "
                "falling back to CPU. Set 'use_cpu' to "
                "True to avoid this warning."
                "\nError encountered was %s" % (name, err))
    try:
        from falkon.cuda import cublas_gpu  # noqa F401
    except Exception as e:
        warnings.warn(get_error_str("cuBLAS", e))
        return False
    try:
        from falkon.cuda import cudart_gpu  # noqa F401
    except Exception as e:
        warnings.warn(get_error_str("cudart", e))
        return False
    try:
        from falkon.cuda import cusolver_gpu  # noqa F401
    except Exception as e:
        warnings.warn(get_error_str("cuSOLVER", e))
        return False
    return True


def decide_keops(opt: KeopsOptions = KeopsOptions()):
    if opt.no_keops:
        return False
    if not hasattr(decide_keops, 'keops_works'):
        try:
            import pykeops  # noqa F401
            # pykeops.clean_pykeops()          # just in case old build files are still present
            # pykeops.test_torch_bindings()
            decide_keops.keops_works = True
        except (ImportError, ModuleNotFoundError):
            warnings.warn("Failed to import PyKeops library; this might lead to "
                          "slower matrix-vector products within FALKON. Please "
                          "install PyKeops and check it works to suppress this warning.")
            decide_keops.keops_works = False
    return decide_keops.keops_works
