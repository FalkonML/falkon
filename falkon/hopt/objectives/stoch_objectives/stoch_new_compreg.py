from typing import Dict, Optional, Tuple

import numpy as np
import torch

import falkon
from falkon import FalkonOptions
from falkon.hopt.objectives.exact_objectives.utils import cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.objectives.stoch_objectives.utils import calc_grads_tensors, init_random_vecs
from falkon.hopt.utils import get_scalar
from falkon.la_helpers import trsm
from falkon.optim import FalkonConjugateGradient
from falkon.preconditioner import FalkonPreconditioner
from falkon.utils.helpers import sizeof_dtype
from falkon.utils.tictoc import Timer

EPS = 5e-5


class StochasticNystromCompReg(HyperoptObjective):
    def __init__(
        self,
        kernel: falkon.kernels.DiffKernel,
        centers_init: torch.Tensor,
        penalty_init: torch.Tensor,
        opt_centers: bool,
        opt_penalty: bool,
        flk_opt: FalkonOptions,
        flk_maxiter: int = 10,
        num_trace_est: int = 20,
        centers_transform: Optional[torch.distributions.Transform] = None,
        pen_transform: Optional[torch.distributions.Transform] = None,
    ):
        super().__init__(kernel, centers_init, penalty_init, opt_centers, opt_penalty, centers_transform, pen_transform)
        self.flk_opt = flk_opt
        self.num_trace_est = num_trace_est
        self.flk_maxiter = flk_maxiter
        self.deterministic_ste = True
        self.gaussian_ste = False
        self.warm_start = True
        self.trace_type = "fast"
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, X, Y):
        loss = stochastic_nystrom_compreg(
            kernel=self.kernel,
            penalty=self.penalty,
            centers=self.centers,
            X=X,
            Y=Y,
            num_estimators=self.num_trace_est,
            deterministic=self.deterministic_ste,
            solve_options=self.flk_opt,
            solve_maxiter=self.flk_maxiter,
            gaussian_random=self.gaussian_ste,
            warm_start=self.warm_start,
            trace_type=self.trace_type,
        )
        self._save_losses(loss)
        return loss

    def predict(self, X):
        if NystromCompRegFn.last_alpha is None:
            raise RuntimeError("Call hp_loss before calling predict.")

        with torch.autograd.no_grad():
            return self.kernel.mmv(X, self.centers, NystromCompRegFn.last_alpha, opt=self.flk_opt)

    def print_times(self):
        NystromCompRegFn.print_times()

    @property
    def last_beta(self):
        return NystromCompRegFn._last_solve_y

    def _save_losses(self, loss):
        self.losses = {
            "compreg_loss": loss.detach(),
        }

    def __repr__(self):
        return (
            f"StochasticNystromCompReg("
            f"kernel={self.kernel}, "
            f"penalty={get_scalar(self.penalty)}, num_centers={self.centers.shape[0]}, "
            f"t={self.num_trace_est}, "
            f"flk_iter={self.flk_maxiter}, det_ste={self.deterministic_ste}, "
            f"gauss_ste={self.gaussian_ste}, warm={self.warm_start}, "
            f"cg_tolerance={self.flk_opt.cg_tolerance}, trace_type={self.trace_type})"
        )


def calc_trace_fwd(
    init_val: torch.Tensor,
    k_mn: Optional[torch.Tensor],
    k_mn_zy: Optional[torch.Tensor],
    kmm_chol: torch.Tensor,
    X: Optional[torch.Tensor],
    t: Optional[int],
    trace_type: str,
):
    """Nystrom kernel trace forward"""
    if trace_type == "ste":
        assert k_mn_zy is not None and t is not None, "Incorrect arguments to trace_fwd"
        solve1 = torch.linalg.solve_triangular(kmm_chol, k_mn_zy[:, :t], upper=False)  # m * t
        solve2 = torch.linalg.solve_triangular(kmm_chol.T, solve1, upper=True).contiguous()  # m * t
        init_val -= solve1.square_().sum(0).mean()
    elif trace_type == "direct":
        assert k_mn is not None, "Incorrect arguments to trace_fwd"
        solve1 = trsm(k_mn, kmm_chol, 1.0, lower=True, transpose=False)  # (M*N)
        solve2 = trsm(solve1, kmm_chol, 1.0, lower=True, transpose=True)  # (M*N)
        init_val -= solve1.square_().sum()
    elif trace_type == "fast":
        assert k_mn_zy is not None and t is not None, "Incorrect arguments to trace_fwd"
        k_subs = k_mn_zy
        assert k_subs.shape == (kmm_chol.shape[0], t), "Shape incorrect"  # m * t
        solve1 = torch.linalg.solve_triangular(kmm_chol, k_subs, upper=False)  # m * t
        solve2 = torch.linalg.solve_triangular(kmm_chol.T, solve1, upper=True).contiguous()  # m * t
        norm = X.shape[0] / t
        init_val -= solve1.square_().sum() * norm
    else:
        raise ValueError("Trace-type %s unknown" % (trace_type))
    return init_val, solve2


def calc_trace_bwd(
    k_mn: Optional[torch.Tensor],
    k_mn_zy: Optional[torch.Tensor],
    solve2: torch.Tensor,
    kmm: torch.Tensor,
    X: Optional[torch.Tensor],
    t: Optional[int],
    trace_type: str,
):
    """Nystrom kernel trace backward pass"""
    if trace_type == "ste":
        assert k_mn_zy is not None and t is not None, "Incorrect arguments to trace_bwd"
        return -(2 * (k_mn_zy[:, :t].mul(solve2)).sum(0).mean() - (solve2 * (kmm @ solve2)).sum(0).mean())
    elif trace_type == "direct":
        assert k_mn is not None, "Incorrect arguments to trace_bwd"
        return -(2 * (k_mn.mul(solve2)).sum() - (solve2 * (kmm @ solve2)).sum())
    elif trace_type == "fast":
        assert k_mn_zy is not None and t is not None and X is not None, "Incorrect arguments to trace_bwd"
        k_subs = k_mn_zy
        norm = X.shape[0] / t
        return -norm * (2 * k_subs.mul(solve2).sum() - (solve2 * (kmm @ solve2)).sum())


def calc_deff_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t, include_kmm_term):
    """Nystrom effective dimension backward"""
    out_deff_bwd = 2 * zy_knm_solve_zy[:t].mean() - zy_solve_knm_knm_solve_zy[:t].mean()
    if include_kmm_term:
        out_deff_bwd -= pen_n * zy_solve_kmm_solve_zy[:t].mean()
    return out_deff_bwd


def calc_dfit_bwd(zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t, include_kmm_term):
    """Nystrom regularized data-fit backward"""
    dfit_bwd = -(2 * zy_knm_solve_zy[t:].sum() - zy_solve_knm_knm_solve_zy[t:].sum())
    if include_kmm_term:
        dfit_bwd += pen_n * zy_solve_kmm_solve_zy[t:].sum()
    return dfit_bwd


# noinspection PyMethodOverriding,PyAbstractClass
class NystromCompRegFn(torch.autograd.Function):
    coef_nm = 40
    _last_solve_z = None
    _last_solve_y = None
    _last_solve_zy = None
    _last_t = None
    last_alpha = None
    iter_prep_times, fwd_times, bwd_times, solve_times, kmm_times, grad_times = [], [], [], [], [], []
    iter_times, num_flk_iters = [], []
    num_nondiff_inputs = 10

    @staticmethod
    def print_times():
        num_times = len(NystromCompRegFn.iter_times)
        print(
            f"Timings: Preparation {np.sum(NystromCompRegFn.iter_prep_times) / num_times:.2f} "
            f"Falkon solve {np.sum(NystromCompRegFn.solve_times) / num_times:.2f} "
            f"(in {np.sum(NystromCompRegFn.num_flk_iters) / num_times:.1f} iters) "
            f"KMM (toCUDA) {np.sum(NystromCompRegFn.kmm_times) / num_times:.2f} "
            f"Forward {np.sum(NystromCompRegFn.fwd_times) / num_times:.2f} "
            f"Backward {np.sum(NystromCompRegFn.bwd_times) / num_times:.2f} "
            f"Grad {np.sum(NystromCompRegFn.grad_times) / num_times:.2f} "
            f"\n\tTotal {np.sum(NystromCompRegFn.iter_times) / num_times:.2f}"
        )
        (
            NystromCompRegFn.iter_prep_times,
            NystromCompRegFn.fwd_times,
            NystromCompRegFn.bwd_times,
            NystromCompRegFn.solve_times,
            NystromCompRegFn.kmm_times,
            NystromCompRegFn.grad_times,
            NystromCompRegFn.iter_times,
            NystromCompRegFn.num_flk_iters,
        ) = ([], [], [], [], [], [], [], [])

    @staticmethod
    def direct_nosplit(
        X: torch.Tensor,
        M: torch.Tensor,
        Y: torch.Tensor,
        penalty: torch.Tensor,
        kmm,
        kmm_chol,
        zy,
        solve_zy,
        zy_solve_kmm_solve_zy,
        kernel: falkon.kernels.DiffKernel,
        t,
        trace_type: str,
        solve_options: falkon.FalkonOptions,
    ):
        k_subs = None
        with Timer(NystromCompRegFn.iter_prep_times), torch.autograd.enable_grad():
            k_mn_zy = kernel.mmv(M, X, zy, opt=solve_options)  # M x (T+P)
            zy_knm_solve_zy = k_mn_zy.mul(solve_zy).sum(0)  # T+P
            if trace_type == "fast":
                rnd_pts = np.random.choice(X.shape[0], size=M.shape[0], replace=False)
                x_subs = X[rnd_pts, :]
                k_subs = kernel(M, x_subs, opt=solve_options)

        # Forward
        dfit_fwd = Y.square().sum().to(M.device)
        deff_fwd = torch.tensor(0, dtype=X.dtype, device=M.device)
        trace_fwd = kernel(X, X, diag=True).sum()
        with Timer(NystromCompRegFn.fwd_times), torch.autograd.no_grad():
            pen_n = penalty * X.shape[0]
            if trace_type == "fast":
                _trace_fwd, solve2 = calc_trace_fwd(
                    init_val=trace_fwd,
                    k_mn=None,
                    k_mn_zy=k_subs,
                    kmm_chol=kmm_chol,
                    t=M.shape[0],
                    trace_type=trace_type,
                    X=X,
                )
            elif trace_type == "ste":
                _trace_fwd, solve2 = calc_trace_fwd(
                    init_val=trace_fwd,
                    k_mn=None,
                    k_mn_zy=k_mn_zy,
                    kmm_chol=kmm_chol,
                    t=t,
                    trace_type=trace_type,
                    X=None,
                )
            else:
                raise NotImplementedError("trace-type %s not implemented." % (trace_type))
            # Nystrom effective dimension forward
            deff_fwd += zy_knm_solve_zy[:t].mean()
            # Data-fit forward
            dfit_fwd -= zy_knm_solve_zy[t:].sum()
            trace_fwd = (_trace_fwd * dfit_fwd) / (pen_n * X.shape[0])
        # Backward
        with Timer(NystromCompRegFn.bwd_times), torch.autograd.enable_grad():
            zy_solve_knm_knm_solve_zy = kernel.mmv(X, M, solve_zy, opt=solve_options).square().sum(0)  # T+P
            pen_n = penalty * X.shape[0]
            # Nystrom effective dimension backward
            deff_bwd = calc_deff_bwd(
                zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t, include_kmm_term=True
            )
            # Data-fit backward
            dfit_bwd = calc_dfit_bwd(
                zy_knm_solve_zy, zy_solve_knm_knm_solve_zy, zy_solve_kmm_solve_zy, pen_n, t, include_kmm_term=True
            )
            # Nystrom kernel trace backward
            if trace_type == "fast":
                trace_bwd = calc_trace_bwd(
                    k_mn=None, k_mn_zy=k_subs, kmm=kmm, X=X, solve2=solve2, t=M.shape[0], trace_type=trace_type
                )
            elif trace_type == "ste":
                trace_bwd = calc_trace_bwd(
                    k_mn=None, k_mn_zy=k_mn_zy, kmm=kmm, X=X, solve2=solve2, t=t, trace_type=trace_type
                )
            else:
                raise NotImplementedError("trace-type %s not implemented." % (trace_type))
            trace_fwd_num = (_trace_fwd * dfit_fwd).detach()
            trace_bwd_num = trace_bwd * dfit_fwd.detach() + _trace_fwd.detach() * dfit_bwd
            trace_den = pen_n * X.shape[0]
            trace_bwd = (trace_bwd_num * trace_den.detach() - trace_fwd_num * trace_den) / (trace_den.detach() ** 2)
            bwd = deff_bwd + dfit_bwd + trace_bwd
            # bwd = dfit_bwd
        return (deff_fwd, dfit_fwd, trace_fwd), bwd

    @staticmethod
    def choose_device_mem(
        data_dev: torch.device, dtype: torch.dtype, solve_options: FalkonOptions
    ) -> Tuple[torch.device, float]:
        if data_dev.type == "cuda":  # CUDA in-core
            from falkon.mmv_ops.utils import _get_gpu_info

            gpu_info = _get_gpu_info(solve_options, slack=solve_options.memory_slack)
            single_gpu_info = [g for g in gpu_info if g.Id == data_dev.index][0]
            avail_mem = single_gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % single_gpu_info.Id)
        elif not solve_options.use_cpu and torch.cuda.is_available():  # CUDA out-of-core
            from falkon.mmv_ops.utils import _get_gpu_info

            gpu_info = _get_gpu_info(solve_options, slack=solve_options.memory_slack)[0]  # TODO: Splitting across gpus
            avail_mem = gpu_info.usable_memory / sizeof_dtype(dtype)
            device = torch.device("cuda:%d" % gpu_info.Id)
        else:  # CPU in-core
            avail_mem = solve_options.max_cpu_mem / sizeof_dtype(dtype)
            device = torch.device("cpu")

        return device, avail_mem

    @staticmethod
    def solve_flk(X, M, Z, ZY, penalty, kernel: falkon.kernels.DiffKernel, solve_options, solve_maxiter, warm_start):
        t = Z.shape[1]

        kernel = kernel.detach()
        penalty_ = penalty.item()
        M_ = M.detach()

        precond = FalkonPreconditioner(penalty_, kernel, solve_options)
        precond.init(M_)

        optim = FalkonConjugateGradient(kernel, precond, solve_options)
        solve_zy_prec = optim.solve(
            X,
            M_,
            ZY,
            penalty_,
            initial_solution=NystromCompRegFn._last_solve_zy,
            max_iter=solve_maxiter,
        )
        if warm_start:
            NystromCompRegFn._last_solve_zy = solve_zy_prec.detach().clone()
            NystromCompRegFn._last_solve_y = NystromCompRegFn._last_solve_zy[:, t:].clone()
        solve_zy = precond.apply(solve_zy_prec)
        NystromCompRegFn.last_alpha = solve_zy[:, t:].detach().clone()
        num_iters = optim.optimizer.num_iter
        return solve_zy, num_iters

    @staticmethod
    def forward(
        ctx,
        kernel: falkon.kernels.DiffKernel,
        deterministic: bool,
        solve_options: FalkonOptions,
        solve_maxiter: int,
        gaussian_random: bool,
        warm_start: bool,
        trace_type: str,
        t: int,
        X: torch.Tensor,
        Y: torch.Tensor,
        penalty: torch.Tensor,
        M: torch.Tensor,
        *kernel_params,
    ):
        if NystromCompRegFn._last_t is not None and NystromCompRegFn._last_t != t:
            NystromCompRegFn._last_solve_y = None
            NystromCompRegFn._last_solve_z = None
            NystromCompRegFn.last_alpha = None
        NystromCompRegFn._last_t = t
        if deterministic:
            torch.manual_seed(12)

        if X.shape[1] < 50:  # If keops need data to stay on CPU
            device, avail_mem = X.device, None
        else:
            # Only device is used
            device, avail_mem = NystromCompRegFn.choose_device_mem(X.device, X.dtype, solve_options)

        with Timer(NystromCompRegFn.iter_times):
            # Initialize hutch trace estimation vectors (t of them)
            Z = init_random_vecs(X.shape[0], t, dtype=X.dtype, device=X.device, gaussian_random=gaussian_random)
            ZY = torch.cat((Z, Y), dim=1)
            M_dev = M.to(device, copy=False).requires_grad_(M.requires_grad)
            with torch.autograd.enable_grad():
                kernel_dev = kernel.to(device)
            penalty_dev = penalty.to(device, copy=False).requires_grad_(penalty.requires_grad)

            with Timer(NystromCompRegFn.solve_times), torch.inference_mode():
                solve_zy, num_flk_iters = NystromCompRegFn.solve_flk(
                    X=X,
                    M=M_dev,
                    Z=Z,
                    ZY=ZY,
                    penalty=penalty_dev,
                    kernel=kernel_dev,
                    solve_options=solve_options,
                    solve_maxiter=solve_maxiter,
                    warm_start=warm_start,
                )
                NystromCompRegFn.num_flk_iters.append(num_flk_iters)

            with Timer(NystromCompRegFn.kmm_times):
                # copy=True to get a normal non-inference tensor
                solve_zy_dev = solve_zy.to(device, copy=True)

                with torch.autograd.enable_grad():
                    kmm = kernel_dev(M_dev, M_dev, opt=solve_options)
                    zy_solve_kmm_solve_zy = (kmm @ solve_zy_dev * solve_zy_dev).sum(0)  # (T+1)
                    # The following should be identical but seems to introduce errors in the bwd pass.
                    # zy_solve_kmm_solve_zy = (kmm_chol.T @ solve_zy_dev).square().sum(0)  # (T+1)
                with torch.autograd.no_grad():
                    mm_eye = torch.eye(M_dev.shape[0], device=device, dtype=M_dev.dtype) * EPS
                    kmm_chol = cholesky(kmm + mm_eye, upper=False, check_errors=False)

            fwd, bwd = NystromCompRegFn.direct_nosplit(
                X=X,
                M=M_dev,
                Y=Y,
                penalty=penalty_dev,
                kmm=kmm,
                kmm_chol=kmm_chol,
                zy=ZY,
                solve_zy=solve_zy_dev,
                zy_solve_kmm_solve_zy=zy_solve_kmm_solve_zy,
                kernel=kernel_dev,
                t=t,
                trace_type=trace_type,
                solve_options=solve_options,
            )
            with Timer(NystromCompRegFn.grad_times):
                grads_ = calc_grads_tensors(
                    inputs=[penalty_dev, M_dev] + list(kernel_dev.diff_params.values()),
                    inputs_need_grad=ctx.needs_input_grad,
                    output=bwd,
                    num_nondiff_inputs=NystromCompRegFn.num_nondiff_inputs,
                    retain_graph=False,
                    allow_unused=False,
                )
                grads = [g if g is None else g.to(X.device) for g in grads_]

        deff_fwd, dfit_fwd, trace_fwd = fwd
        ctx.grads = grads
        # print(f"Stochastic: D-eff {deff_fwd:.3e} Data-Fit {dfit_fwd:.3e} Trace {trace_fwd:.3e}")
        return (deff_fwd + dfit_fwd + trace_fwd).to(X.device)

    @staticmethod
    def backward(ctx, out):
        grads_out = []
        for g in ctx.grads:
            if g is not None:
                g = g * out
            grads_out.append(g)
        return tuple(grads_out)

    @staticmethod
    def grad_check():
        torch.manual_seed(3)
        X = torch.randn(50, 6, dtype=torch.float64)
        w = torch.randn(X.shape[1], 1, dtype=torch.float64)
        Y = X @ w
        M = X[:10].clone().detach().requires_grad_()
        s = torch.tensor([10.0], dtype=X.dtype).requires_grad_()
        p = torch.tensor(1e-2, dtype=X.dtype).requires_grad_()
        torch.autograd.gradcheck(
            lambda sigma, pen, centers: NystromCompRegFn.apply(
                sigma,  # kernel_args
                pen,  # penalty
                centers,  # M
                X,  # X
                Y,  # Y
                20,  # t
                True,  # deterministic
                FalkonOptions(),  # solve_options
                30,  # solve_maxiter
                False,  # gaussian_random
                True,  # use_stoch_trace
                False,
            ),  # warm_start
            (s, p, M),
        )


def stochastic_nystrom_compreg(
    kernel,
    penalty,
    centers,
    X,
    Y,
    num_estimators,
    deterministic,
    solve_options,
    solve_maxiter,
    gaussian_random,
    warm_start=True,
    trace_type="ste",
):
    return NystromCompRegFn.apply(
        kernel,
        deterministic,
        solve_options,
        solve_maxiter,
        gaussian_random,
        warm_start,
        trace_type,
        num_estimators,
        X,
        Y,
        penalty,
        centers,
        *kernel.diff_params.values(),
    )
