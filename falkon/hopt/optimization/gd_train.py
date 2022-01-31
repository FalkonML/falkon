import time
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import torch

from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.optimization.reporting import pred_reporting, EarlyStop, epoch_bookkeeping

__all__ = [
    "train_complexity_reg",
    "train_complexity_reg_mb",
]


def hp_grad(model: HyperoptObjective, *loss_terms, accumulate_grads=True, verbose=True):
    grads = []
    hparams = list(model.parameters())
    if verbose:
        for loss in loss_terms:
            grads.append(torch.autograd.grad(loss, hparams, retain_graph=True, allow_unused=True))
    else:
        loss = reduce(torch.add, loss_terms)
        grads.append(torch.autograd.grad(loss, hparams, retain_graph=False))

    if accumulate_grads:
        for g in grads:
            for i in range(len(hparams)):
                hp = hparams[i]
                if hp.grad is None:
                    hp.grad = torch.zeros_like(hp)
                if g[i] is not None:
                    hp.grad += g[i]
    return grads


def create_optimizer(opt_type: str, model: HyperoptObjective, learning_rate: float):
    center_lr_div = 1
    schedule = None
    named_params = dict(model.named_parameters())
    print("Creating optimizer with the following parameters:")
    for k, v in named_params.items():
        print(f"\t{k} : {v.shape}")
    if opt_type == "adam":
        if 'penalty' not in named_params:
            opt_modules = [
                {"params": named_params.values(), 'lr': learning_rate}
            ]
        else:
            opt_modules = []
            if 'sigma' in named_params:
                opt_modules.append({"params": named_params['sigma'], 'lr': learning_rate})
            if 'penalty' in named_params:
                opt_modules.append({"params": named_params['penalty'], 'lr': learning_rate})
            if 'centers' in named_params:
                opt_modules.append({
                    "params": named_params['centers'], 'lr': learning_rate / center_lr_div})
        opt_hp = torch.optim.Adam(opt_modules)
        # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hp, factor=0.5, patience=1)
        # schedule = torch.optim.lr_scheduler.MultiStepLR(opt_hp, [2, 10, 40], gamma=0.5)
        schedule = torch.optim.lr_scheduler.StepLR(opt_hp, 200, gamma=0.1)
    elif opt_type == "sgd":
        opt_hp = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_type == "lbfgs":
        if model.losses_are_grads:
            raise ValueError("L-BFGS not valid for model %s" % (model))
        opt_hp = torch.optim.LBFGS(model.parameters(), lr=learning_rate,
                                   history_size=100, )
    elif opt_type == "rmsprop":
        opt_hp = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Optimizer type %s not recognized" % (opt_type))

    return opt_hp, schedule


def train_complexity_reg(
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        Xts: torch.Tensor,
        Yts: torch.Tensor,
        model: HyperoptObjective,
        err_fn,
        learning_rate: float,
        num_epochs: int,
        cuda: bool,
        loss_every: int,
        early_stop_epochs: int,
        cgtol_decrease_epochs: Optional[int],
        optimizer: str,
        retrain_nkrr: bool = False,
) -> List[Dict[str, float]]:
    if cuda:
        Xtr, Ytr, Xts, Yts = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    opt_hp, schedule = create_optimizer(optimizer, model, learning_rate)
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer.")

    logs = []
    cum_time = 0
    with torch.autograd.profiler.profile(enabled=False) as prof:
        for epoch in range(num_epochs):
            t_start = time.time()

            def closure():
                opt_hp.zero_grad()
                loss = model(Xtr, Ytr)
                loss.backward()
                return float(loss)
            try:
                opt_hp.step(closure)
            except RuntimeError as e:
                if "Cholesky" not in str(e):
                    raise e
                print(f"Cholesky failure at epoch {epoch}. Exiting optimization!")
                break

            cum_time += time.time() - t_start
            try:
                epoch_bookkeeping(epoch=epoch, model=model, data={'Xtr': Xtr, 'Ytr': Ytr, 'Xts': Xts, 'Yts': Yts},
                                  err_fn=err_fn, loss_every=loss_every,
                                  early_stop_patience=early_stop_epochs, schedule=schedule,
                                  minibatch=None, logs=logs, cum_time=cum_time,
                                  accuracy_increase_patience=cgtol_decrease_epochs)
            except EarlyStop as e:
                print(e)
                break
        torch.cuda.empty_cache()
    if prof is not None:
        print(prof.key_averages().table())
    if retrain_nkrr:
        print(f"Final retrain after {num_epochs} epochs:")
        pred_dict = pred_reporting(
            model=model, Xtr=Xtr, Ytr=Ytr, Xts=Xts, Yts=Yts,
            err_fn=err_fn, epoch=num_epochs, cum_time=cum_time,
            resolve_model=True)
        logs.append(pred_dict)

    return logs


def train_complexity_reg_mb(
        Xtr: torch.Tensor,
        Ytr: torch.Tensor,
        Xts: torch.Tensor,
        Yts: torch.Tensor,
        model: HyperoptObjective,
        err_fn,
        learning_rate: float,
        num_epochs: int,
        cuda: bool,
        loss_every: int,
        early_stop_epochs: int,
        cgtol_decrease_epochs: Optional[int],
        optimizer: str,
        minibatch: int,
        retrain_nkrr: bool = False,
) -> List[Dict[str, float]]:
    Xtrc, Ytrc, Xtsc, Ytsc = Xtr, Ytr, Xts, Yts
    if cuda:
        Xtrc, Ytrc, Xtsc, Ytsc = Xtr.cuda(), Ytr.cuda(), Xts.cuda(), Yts.cuda()
    opt_hp, schedule = create_optimizer(optimizer, model, learning_rate)
    print(f"Starting hyperparameter optimization on model {model}.")
    print(f"Will run for {num_epochs} epochs with {opt_hp} optimizer, "
          f"mini-batch size {minibatch}.")

    logs = []
    cum_time = 0
    mb_indices = np.arange(Xtr.shape[0])
    for epoch in range(num_epochs):
        t_start = time.time()
        np.random.shuffle(mb_indices)
        for mb_start in range(0, Xtr.shape[0], minibatch):
            Xtr_batch = (Xtr[mb_indices[mb_start: mb_start + minibatch], :]).contiguous()
            Ytr_batch = (Ytr[mb_indices[mb_start: mb_start + minibatch], :]).contiguous()
            if cuda:
                Xtr_batch, Ytr_batch = Xtr_batch.cuda(), Ytr_batch.cuda()

            opt_hp.zero_grad()
            loss = model(Xtr_batch, Ytr_batch)
            loss.backward()
            opt_hp.step()

        cum_time += time.time() - t_start
        try:
            epoch_bookkeeping(epoch=epoch, model=model, data={'Xtr': Xtrc, 'Ytr': Ytrc, 'Xts': Xtsc, 'Yts': Ytsc},
                              err_fn=err_fn, loss_every=loss_every,
                              early_stop_patience=early_stop_epochs, schedule=schedule,
                              minibatch=minibatch, logs=logs, cum_time=cum_time,
                              accuracy_increase_patience=cgtol_decrease_epochs)
        except EarlyStop as e:
            print(e)
            break
    if retrain_nkrr:
        print(f"Final retrain after {num_epochs} epochs:")
        pred_dict = pred_reporting(
            model=model, Xtr=Xtrc, Ytr=Ytrc, Xts=Xtsc, Yts=Ytsc,
            err_fn=err_fn, epoch=num_epochs, cum_time=cum_time,
            resolve_model=True)
        logs.append(pred_dict)

    return logs
