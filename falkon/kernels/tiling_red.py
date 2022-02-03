from functools import partial
from typing import List, Optional, Tuple
from copy import copy
import warnings

import torch
from dataclasses import dataclass

from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import LoadKeOps
from pykeops.common.operations import preprocess, postprocess
from pykeops.torch.half2_convert import preprocess_half2, postprocess_half2
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases, get_optional_flags
from pykeops.common.utils import axis2cat
from pykeops.torch import default_dtype, include_dirs

from falkon import FalkonOptions
from falkon.mmv_ops.keops import ArgsFmmv, _estimate_split
from falkon.mmv_ops.utils import _start_wait_processes
from falkon.utils import devices
from falkon.utils.helpers import check_same_device, calc_gpu_block_sizes, sizeof_dtype


@dataclass
class KeopsTags:
    tagCPUGPU: int
    tag1D2D: int
    tagHostDevice: int


def _single_gpu_method(proc_idx, queue, device_id):
    a: ArgsFmmv = queue.get()
    X1 = a.X1
    X2 = a.X2
    v = a.v
    oout = a.out
    other_vars = a.other_vars
    fn = a.function
    R = a.gpu_ram

    N, D = X1.shape
    M = X2.shape[0]
    T = v.shape[1]

    # Second round of subdivision (only if necessary due to RAM constraints)
    n, m = _estimate_split(N, M, D, T, R, sizeof_dtype(X1.dtype))

    # Process the two rounds of splitting with a nested loop.
    for mi in range(0, M, m):
        ml = min(m, M - mi)
        if ml != M and mi > 0:  # Then we must create a temporary output array
            out = torch.empty_like(oout)
        else:
            out = oout

        cX2 = X2[mi:mi + ml, :]
        cv = v[mi:mi + ml, :]

        for ni in range(0, N, n):
            nl = min(n, N - ni)
            cX1 = X1[ni:ni + nl, :]
            cout = out[ni: ni + nl, :]

            variables = [cX1, cX2, cv] + other_vars
            fn(nl,    # nx
               ml,    # ny
               cout,  # out
               *variables)
        torch.cuda.synchronize(device_id)
        if ml != M and mi > 0:
            oout.add_(out)

    return oout


def load_keops_fn(formula, aliases, backend, dtype, optional_flags, rec_multVar_highdim, out, *args):
    flags = copy(optional_flags)
    if rec_multVar_highdim is not None:
        flags.append('-DMULT_VAR_HIGHDIM=1')
    fn = LoadKeOps(formula, aliases, dtype, 'torch', flags, include_dirs).import_module()
    tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args, output=out)
    tags = KeopsTags(tagCPUGPU=tagCPUGPU, tag1D2D=tag1D2D, tagHostDevice=tagHostDevice)
    return fn, tags


def run_keops_formula(formula, aliases, backend, dtype, device, ranges, nx, ny, optional_flags, rec_multVar_highdim, out, *args):
    myconv, tags = load_keops_fn(formula, aliases, backend, dtype, optional_flags, rec_multVar_highdim, out, *args)
    device_id = device.index or -1
    if out is None:
        out = myconv.genred_pytorch(tags.tagCPUGPU, tags.tag1D2D, tags.tagHostDevice, device_id, ranges, nx, ny, *args)
    else:
        myconv.genred_pytorch_out(tags.tagCPUGPU, tags.tag1D2D, tags.tagHostDevice, device_id, ranges, nx, ny, out, *args)
    return out


def run_mmv_formula(formula: str,
                    aliases: List[str],
                    backend: str,
                    dtype: str,
                    ranges: tuple,
                    optional_flags: List[str],
                    rec_multVar_highdim,
                    nx,
                    ny,
                    data_device: torch.device,
                    X1: torch.Tensor,
                    X2: torch.Tensor,
                    v: torch.Tensor,
                    out: torch.Tensor,
                    other_vars: Tuple[torch.Tensor, ...],
                    opt: FalkonOptions):
    other_vars = list(other_vars)
    variables = [X1, X2, v] + other_vars
    myconv, tags = load_keops_fn(formula, aliases, backend, dtype, optional_flags, rec_multVar_highdim, out, *variables)

    if backend.startswith("GPU") and data_device.type == 'cpu':
        # Info about GPUs
        ram_slack = 0.7  # slack is high due to imprecise memory usage estimates
        gpu_info = [v for k, v in devices.get_device_info(opt).items() if k >= 0]
        gpu_ram = [
            min((g.free_memory - 300 * 2 ** 20) * ram_slack, opt.max_gpu_mem * ram_slack)
            for g in gpu_info
        ]
        block_sizes = calc_gpu_block_sizes(gpu_info, X1.shape[0])

        # Create queues
        subproc_args = []  # Arguments passed to each subprocess
        for i in range(len(gpu_info)):
            # First round of subdivision
            bwidth = block_sizes[i + 1] - block_sizes[i]
            if bwidth <= 0:
                continue
            fn = partial(myconv.genred_pytorch_out,
                         tags.tagCPUGPU,
                         tags.tag1D2D,
                         tags.tagHostDevice,
                         gpu_info[i].Id,
                         ranges,
                         )
            subproc_args.append((ArgsFmmv(
                X1=X1.narrow(0, block_sizes[i], bwidth),
                X2=X2,
                v=v,
                out=out.narrow(0, block_sizes[i], bwidth),
                other_vars=other_vars,
                function=fn,
                backend=backend,
                gpu_ram=gpu_ram[i]
            ), gpu_info[i].Id))
        _start_wait_processes(_single_gpu_method, subproc_args)
    else:  # Run on CPU or GPU with CUDA inputs
        device_idx = data_device.index or -1
        fn = partial(myconv.genred_pytorch_out, tags.tagCPUGPU, tags.tag1D2D,
                     tags.tagHostDevice, device_idx, ranges, nx, ny)
        out = fn(out, *variables)

    return out, myconv


# noinspection PyMethodOverriding
class TilingGenredAutograd(torch.autograd.Function):
    """
    This class is the entry point to pytorch auto grad engine.
    """
    NUM_NON_GRAD_ARGS = 11

    @staticmethod
    def forward(ctx,
                formula: str,
                aliases: List[str],
                backend: str,
                dtype: str,
                ranges: Optional[tuple],
                optional_flags: List[str],
                rec_multVar_highdim,
                nx,
                ny,
                opt: FalkonOptions,
                out: Optional[torch.Tensor],
                X1: torch.Tensor,
                X2: torch.Tensor,
                v: torch.Tensor,
                *args: torch.Tensor):
        # Context variables: save everything to compute the gradient:
        ctx.optional_flags = optional_flags.copy()

        tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)
        if not check_same_device(out, *args):
            raise ValueError("[KeOps] Input and output arrays must be located on the same device.")
        device = X1.device

        # N.B.: KeOps C++ expects contiguous data arrays
        if not check_all_contig(X1, X2, v, *args):
            warnings.warn(
                "[pyKeOps,TilingGenRed] At least one of the input tensors is not contiguous. "
                "Consider using contiguous data arrays to avoid unnecessary copies.")
            X1, X2, v, *args = ensure_all_contig(X1, X2, v, *args)
        if out is None:
            # noinspection PyArgumentList
            out = torch.empty(X1.shape[0], v.shape[1], dtype=X1.dtype, device=device,
                              pin_memory=(device.type == "cpu" and tagCPUGPU == 1))
        if ranges is None:
            ranges = ()  # To keep the same type

        out, conv_module = run_mmv_formula(formula=formula,
                                           aliases=aliases,
                                           backend=backend,
                                           dtype=dtype,
                                           ranges=ranges,
                                           optional_flags=optional_flags,
                                           rec_multVar_highdim=rec_multVar_highdim,
                                           nx=nx,
                                           ny=ny,
                                           data_device=device,
                                           X1=X1,
                                           X2=X2,
                                           v=v,
                                           out=out,
                                           other_vars=args,
                                           opt=opt
                                           )

        # Context variables: save everything to compute the gradient:
        ctx.formula = formula
        ctx.aliases = aliases
        ctx.backend = backend
        ctx.dtype = dtype
        ctx.ranges = ranges
        ctx.myconv = conv_module
        ctx.rec_multVar_highdim = rec_multVar_highdim
        ctx.device = device
        ctx.nx = nx
        ctx.ny = ny

        # relying on the 'ctx.saved_variables' attribute is necessary
        # if you want to be able to differentiate the output of the backward once again.
        # It helps pytorch to keep track of 'who is who'.
        ctx.save_for_backward(out, X1, X2, v, *args)
        return out

    @staticmethod
    def _check_bw_red_support(formula):
        not_supported = ["Min_ArgMin_Reduction", "Min_Reduction",
                         "Max_ArgMax_Reduction", "Max_Reduction",
                         "KMin_ArgKMin_Reduction", "KMin_Reduction"]
        for red in not_supported:
            if formula.startswith(red):
                raise NotImplementedError("As of today, KeOps does not support "
                                          + "backpropagation through the " + red + " reduction. "
                                          + "Adding this feature to LazyTensors is on the cards "
                                          + "for future releases... But until then, you may want "
                                          + "to consider extracting the relevant integer indices "
                                          + "with a '.argmin()', '.argmax()' or '.argKmin()' reduction "
                                          + "before using PyTorch advanced indexing to create a fully-differentiable "
                                          + "tensor containing the relevant 'minimal' values.")

    @staticmethod
    def backward(ctx, G) -> Tuple[Optional[torch.Tensor], ...]:
        formula = ctx.formula
        aliases = ctx.aliases
        backend = ctx.backend
        dtype = ctx.dtype
        ranges = ctx.ranges
        optional_flags = ctx.optional_flags
        device: torch.device = ctx.device
        myconv = ctx.myconv
        nx = ctx.nx
        ny = ctx.ny
        args = ctx.saved_tensors[1:]  # Unwrap the saved variables
        nargs = len(args)
        result = ctx.saved_tensors[0].detach()

        TilingGenredAutograd._check_bw_red_support(formula)

        # If formula takes 5 variables (numbered from 0 to 4), then the gradient
        # wrt. the output, G, should be given as a 6-th variable (numbered 5),
        # with the same dim-cat as the formula's output.
        eta = f'Var({nargs},{myconv.dimout},{myconv.tagIJ})'
        # there is also a new variable for the formula's output
        resvar = f'Var({nargs+1},{myconv.dimout},{myconv.tagIJ})'

        # convert to contiguous:
        G = G.contiguous()
        grads = []  # list of gradients wrt. args;

        for var_ind, (sig, arg_ind) in enumerate(zip(aliases, args)):  # Run through the arguments
            # If the current gradient is to be discarded immediatly...
            if not ctx.needs_input_grad[var_ind + TilingGenredAutograd.NUM_NON_GRAD_ARGS]:  # because of (formula, aliases, backend, dtype, device_id, ranges, accuracy_flags, out)
                grads.append(None)  # Don't waste time computing it.
            else:
                # Otherwise, the current gradient is really needed by the user:
                # adding new aliases is way too dangerous if we want to compute
                # second derivatives, etc. So we make explicit references to Var<ind,dim,cat> instead.
                # New here (Joan) : we still add the new variables to the list of "aliases" (without
                # giving new aliases for them) these will not be used in the C++ code,
                # but are useful to keep track of the actual variables used in the formula
                _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
                var = f'Var({pos},{dim},{cat})'   # V
                formula_g = f'Grad_WithSavedForward({formula},{var},{eta},{resvar})'  # Grad<F,V,G,R>
                aliases_g = aliases + [eta, resvar]
                args_g = args + (G, result)  # Don't forget the gradient to backprop !

                # For a reduction of the type sum(F*b), with b a variable, and if we require the gradient
                # with respect to b, the gradient will be of same type sum(F*eta). So we set again rec_multVar option
                # in this case.
                if pos == ctx.rec_multVar_highdim:
                    rec_multVar_highdim = nargs  # nargs is the position of variable eta.
                else:
                    rec_multVar_highdim = None

                if cat == 2:  # we're referring to a parameter, so we'll have to sum both wrt 'i' and 'j'
                    # WARNING !! : here we rely on the implementation of DiffT in files in folder keops/core/formulas/reductions
                    # if tagI==cat of V is 2, then reduction is done wrt j, so we need to further sum output wrt i
                    grad = run_keops_formula(formula_g, aliases_g, backend, dtype, device, ranges, nx, ny, optional_flags, rec_multVar_highdim, None, *args_g)
                    # Then, sum 'grad' wrt 'i' :
                    # I think that '.sum''s backward introduces non-contiguous arrays,
                    # and is thus non-compatible with GenredAutograd: grad = grad.sum(0)
                    # We replace it with a 'handmade hack' :
                    # grad = torch.ones(1, grad.shape[0]).type_as(grad.data) @ grad
                    # grad = grad.view(-1)
                    grad = (1. * grad).sum(-2)
                    dims_to_collapse = tuple(
                        i for (i, (x, y)) in enumerate(zip(arg_ind.shape[:-1], grad.shape[:-1])) if
                        x < y)

                else:
                    grad = run_keops_formula(formula_g, aliases_g, backend, dtype, device, ranges, nx, ny, optional_flags, rec_multVar_highdim, None, *args_g)

                    # N.B.: 'grad' is always a full [A, .., B, M, D] or [A, .., B, N, D] or [A, .., B, D] tensor,
                    #       whereas 'arg_ind' may have some broadcasted batched dimensions.
                    #       Before returning our gradient, we must collapse 'grad' with a .sum() operation,
                    #       which is the adjoint of the good old "repmat" that could have been used
                    #       to emulate the batch broadcasting.
                    dims_to_collapse = tuple(
                        i for (i, (x, y)) in enumerate(zip(arg_ind.shape[:-2], grad.shape[:-2])) if
                        x < y)

                if dims_to_collapse != ():
                    grad = (1. * grad).sum(dims_to_collapse, keepdim=True)
                grad = grad.reshape(arg_ind.shape)  # The gradient should have the same shape as the input!
                grads.append(grad)

        # Grads wrt. formula, aliases, backend, dtype, ranges, accuracy_flags, out, *args
        # Where *args is X1, X2, v, other_vars
        return tuple([None] * TilingGenredAutograd.NUM_NON_GRAD_ARGS + grads)


class TilingGenred():
    def __init__(self, formula, aliases, reduction_op='Sum', axis=0, dtype=default_dtype,
                 opt_arg=None, enable_chunks=True, rec_multVar_highdim=None, optional_flags=None,
                 formula2=None, cuda_type=None, dtype_acc="auto", use_double_acc=False,
                 sum_scheme="auto", opt: FalkonOptions = None):
        if optional_flags is None:
            optional_flags = []
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type
        self.reduction_op = reduction_op
        reduction_op_internal, formula2 = preprocess(reduction_op, formula2)
        self.optional_flags = optional_flags + get_optional_flags(reduction_op_internal, dtype_acc, use_double_acc, sum_scheme, dtype, enable_chunks)

        str_opt_arg = ',' + str(opt_arg) if opt_arg else ''
        str_formula2 = ',' + formula2 if formula2 else ''

        self.formula = reduction_op_internal + '_Reduction(' + formula + str_opt_arg + ',' + str(
            axis2cat(axis)) + str_formula2 + ')'
        self.aliases = complete_aliases(self.formula, list(aliases))  # just in case the user provided a tuple
        self.dtype = dtype
        self.axis = axis
        self.opt_arg = opt_arg
        self.opt = opt
        self.rec_multVar_highdim = rec_multVar_highdim

    def __call__(self, *args, out=None, backend='auto', ranges=None):
        nx, ny = get_sizes(self.aliases, *args)
        nout, nred = (nx, ny) if self.axis == 1 else (ny, nx)

        if "Arg" in self.reduction_op:
            # when using Arg type reductions,
            # if nred is greater than 16 millions and dtype=float32, the result is not reliable
            # because we encode indices as floats, so we raise an exception ;
            # same with float16 type and nred>2048
            if nred > 1.6e7 and self.dtype in ("float32", "float"):
                raise ValueError(
                    'size of input array is too large for Arg type reduction with single precision. Use double precision.')
            elif nred > 2048 and self.dtype in ("float16", "half"):
                raise ValueError(
                    'size of input array is too large for Arg type reduction with float16 dtype..')

        if self.dtype in ('float16', 'half'):
            args, ranges, tag_dummy, N = preprocess_half2(
                args, self.aliases, self.axis, ranges, nx, ny)
        out = TilingGenredAutograd.apply(
            self.formula,
            self.aliases,
            backend,
            self.dtype,
            ranges,
            self.optional_flags,
            self.rec_multVar_highdim,
            nx,
            ny,
            self.opt,
            out,
            *args
        )

        if self.dtype in ('float16', 'half'):
            out = postprocess_half2(out, tag_dummy, self.reduction_op, N)

        return postprocess(out, "torch", self.reduction_op, nout, self.opt_arg, self.dtype)


def check_all_contig(*args):
    for a in args:
        if not a.is_contiguous():
            return False
    return True


def ensure_all_contig(*args):
    for a in args:
        if not a.is_contiguous():
            a = a.contiguous()
        yield a
