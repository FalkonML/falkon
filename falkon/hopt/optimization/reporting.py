import dataclasses
from typing import Optional, Dict
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from falkon import FalkonOptions, InCoreFalkon, Falkon
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar

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


class EarlyStop(Exception):
    def __init__(self, msg):
        super(EarlyStop, self).__init__(msg)


def report_losses(losses, loss_names, step) -> Dict[str, float]:
    assert len(losses) == len(
        loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    writer = get_writer()
    report_str = "LOSSES: "
    report_dict = {}
    loss_sum = 0
    for loss, loss_name in zip(losses, loss_names):
        _loss = get_scalar(loss)
        # Report the value of the loss
        writer.add_scalar(f'optim/{loss_name}', _loss, step)
        report_str += f"{loss_name}: {_loss:.3e} - "
        report_dict[f"loss_{loss_name}"] = _loss
        loss_sum += _loss
    if len(losses) > 1:
        report_str += f"tot: {loss_sum:.3e}"
    report_dict["loss"] = loss_sum
    print(report_str, flush=True)
    return report_dict


def report_hps(named_hparams, step) -> Dict[str, float]:
    writer = get_writer()
    report_dict = {}
    for hp_name, hp_val in named_hparams.items():
        hp_val_ = get_scalar(hp_val)
        # Report the hparam value
        writer.add_scalar(f'hparams/{hp_name}', hp_val_, step)
        report_dict[f"hp_{hp_name}"] = hp_val_
    return report_dict


def report_grads(named_hparams, grads, losses, loss_names, step) -> Dict[str, float]:
    assert len(losses) == len(
        loss_names), f"Found {len(losses)} losses and {len(loss_names)} loss-names."
    assert len(grads) == len(losses), f"Found {len(grads)} grads and {len(losses)} losses."
    writer = get_writer()
    report_dict = {}
    for i in range(len(grads)):
        for j, (hp_name, hp_val) in enumerate(named_hparams.items()):
            grad_ = get_scalar(grads[i][j])
            # Report the gradient of a specific loss wrt a specific hparam
            writer.add_scalar(f'grads/{hp_name}/{loss_names[i]}', grad_, step)
            report_dict[f"grad_{hp_name}_{loss_names[i]}"] = grad_
    return report_dict


def grad_loss_reporting(named_hparams, grads, losses, loss_names, verbose, step, losses_are_grads):
    report_dict = {}
    # Losses
    if not losses_are_grads and losses is not None:
        report_dict.update(report_losses(losses, loss_names, step))
    # Hyperparameters
    report_dict.update(report_hps(named_hparams, step))
    # Gradients
    if verbose and grads is not None:
        report_dict.update(report_grads(named_hparams, grads, losses, loss_names, step))
    return report_dict


def pred_reporting(model: HyperoptObjective,
                   Xts: torch.Tensor, Yts: torch.Tensor,
                   Xtr: torch.Tensor, Ytr: torch.Tensor,
                   err_fn: callable,
                   epoch: int,
                   cum_time: float,
                   Xval: Optional[torch.Tensor] = None,
                   Yval: Optional[torch.Tensor] = None,
                   resolve_model: bool = False,
                   mb_size: Optional[int] = None,
                   ) -> Dict[str, float]:
    writer = get_writer()
    model.eval()
    sigma, penalty, centers = model.sigma, model.penalty, model.centers

    if resolve_model:
        flk_opt = FalkonOptions(use_cpu=not torch.cuda.is_available(), cg_tolerance=1e-4,
                                cg_epsilon_32=1e-6)
        from falkon.kernels import GaussianKernel
        from falkon.center_selection import FixedSelector
        kernel = GaussianKernel(sigma.detach().flatten(), flk_opt)
        center_selector = FixedSelector(centers.detach())
        flk_cls = InCoreFalkon if Xtr.is_cuda else Falkon
        flk_model = flk_cls(kernel, penalty.item(), M=centers.shape[0],
                            center_selection=center_selector, maxiter=100,
                            seed=1312, error_fn=err_fn, error_every=None, options=flk_opt)
        Xtr_full, Ytr_full = Xtr, Ytr
        if Xval is not None and Yval is not None:
            Xtr_full, Ytr_full = torch.cat((Xtr, Xval), dim=0), torch.cat((Ytr, Yval), dim=0)
        warm_start = None
        if hasattr(model, "last_beta") and model.last_beta is not None:
            warm_start = model.last_beta.to(Xtr_full.device)
        flk_model.fit(Xtr_full, Ytr_full, warm_start=warm_start)  # , Xts, Yts)
        model = flk_model

    # Predict in mini-batches
    test_preds, train_preds = [], []
    c_mb_size = mb_size or Xts.shape[0]
    for i in range(0, Xts.shape[0], c_mb_size):
        test_preds.append(model.predict(Xts[i: i + c_mb_size]).detach().cpu())
    c_mb_size = mb_size or Xtr.shape[0]
    for i in range(0, Xtr.shape[0], c_mb_size):
        train_preds.append(model.predict(Xtr[i: i + c_mb_size]).detach().cpu())
    test_preds = torch.cat(test_preds, dim=0)
    train_preds = torch.cat(train_preds, dim=0)
    test_err, err_name = err_fn(Yts.detach().cpu(), test_preds)
    train_err, err_name = err_fn(Ytr.detach().cpu(), train_preds)
    out_str = (f"Epoch {epoch} ({cum_time:5.2f}s) - "
               f"Sigma {get_scalar(sigma):.3f} - Penalty {get_scalar(penalty):.2e} - "
               f"Tr  {err_name} = {train_err:9.7f} - "
               f"Ts  {err_name} = {test_err:9.7f}")
    writer.add_scalar(f'error/{err_name}/train', train_err, epoch)
    writer.add_scalar(f'error/{err_name}/test', test_err, epoch)

    if Xval is not None and Yval is not None:
        val_preds = model.predict(Xval).detach().cpu()
        val_err, err_name = err_fn(Yval.detach().cpu(), val_preds)
        out_str += f" - Val {err_name} = {val_err:6.4f}"
        writer.add_scalar(f'error/{err_name}/val', val_err, epoch)
    print(out_str, flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        f"train_{err_name}": train_err,
        f"test_{err_name}": test_err,
        "train_error": train_err,
        "test_error": test_err,
        "cum_time": cum_time,
    }


def epoch_bookkeeping(
        epoch: int,
        model: HyperoptObjective,
        data: Dict[str, torch.Tensor],
        err_fn,
        grads,
        losses,
        loss_every: int,
        early_stop_patience: Optional[int],
        accuracy_increase_patience: Optional[int],
        schedule,
        minibatch: Optional[int],
        logs: list,
        cum_time: float,
        verbose):
    Xtr, Ytr, Xts, Yts = data['Xtr'], data['Ytr'], data['Xts'], data['Yts']

    loss_dict = grad_loss_reporting(model.named_parameters(), grads, losses, model.loss_names,
                                    verbose, epoch, losses_are_grads=model.losses_are_grads)
    if epoch != 0 and (epoch + 1) % loss_every == 0:
        pred_dict = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=err_fn, epoch=epoch, cum_time=cum_time,
            resolve_model=True, mb_size=minibatch)
        if hasattr(model, "print_times"):
            model.print_times()
        loss_dict.update(pred_dict)
    logs.append(loss_dict)
    # Learning rate schedule
    if schedule is not None:
        if isinstance(schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if 'train_error' in loss_dict:
                schedule.step(loss_dict['train_error'])
        else:
            schedule.step()
    # Early stop if no training-error improvement in the past `early_stop_patience` epochs.
    if early_stop_patience is not None and len(logs) >= early_stop_patience:
        if "train_error" in logs[-1]:
            past_errs = []
            past_logs = logs[-early_stop_patience:]  # Last n logs from most oldest to most recent
            for plog in past_logs:
                if 'train_error' in plog:
                    past_errs.append(abs(plog['train_error']))
            if np.argmin(past_errs) == 0:  # The minimal error in the oldest log
                raise EarlyStop(f"Early stopped at epoch {epoch} with past errors: {past_errs}.")
    from falkon.hopt.objectives.stoch_objectives.stoch_new_compreg import StochasticDeffPenFitTr
    if (accuracy_increase_patience is not None and
            len(logs) >= accuracy_increase_patience and
            isinstance(model, StochasticDeffPenFitTr)):
        if "train_error" in logs[-1]:
            past_errs = []
            for plog in logs[::-1]:  # Traverse logs in reverse order
                if 'train_error' in plog:
                    past_errs.append(abs(plog['train_error']))
                if len(past_errs) >= accuracy_increase_patience:
                    break
            print("Past errors: ", past_errs)
            if len(past_errs) >= accuracy_increase_patience:
                if np.argmin(past_errs) == (len(past_errs) - 1):  # The minimal error in the oldest log
                    cur_acc = model.flk_opt.cg_tolerance
                    new_acc = cur_acc / 10
                    print("INFO: Changing tolerance to %e" % (new_acc))
                    new_opt = dataclasses.replace(model.flk_opt, cg_tolerance=new_acc)
                    model.flk_opt = new_opt
