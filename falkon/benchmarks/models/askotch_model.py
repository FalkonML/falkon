import torch
import pykeops
from pykeops.config import gpu_available
from fast_krr.models import FullKRR
from fast_krr.opts import ASkotchV2
from pykeops.torch import LazyTensor

import time 

from tqdm import trange

class ASkotchWrapper:
    
    
#    def __init__(self, opt, num_iter, log_every = 1):
    def __init__(self, Xtr, Ytr, Xts, Yts, block_size, precond_params, kernel_params, lam, task, num_iter, device, log_every=1):

        w0 = torch.zeros((Xtr.shape[0], ), device=device)
        model = FullKRR(Xtr, Ytr, Xts, Yts, kernel_params=kernel_params, Ktr_needed=True, lambd=lam, task=task, w0=w0, device=device)

        self.opt = ASkotchV2(model=model, block_sz=block_size, precond_params=precond_params)

#        self.opt = opt
        self.kern_fn = self.opt.model._get_kernel_fn()
        self.num_iter = num_iter
        self.log_every = log_every

    def fit(self, Xtr, Ytr, Xts, Yts, err_fn):
        
#        ker_fun = self.opt.model._get_kernel_fn()
#        ft_time = time.time()
#        tr_errors, te_errors, it_times = [], [], []
        for i in trange(1, self.num_iter + 1, desc="Optimization progress"):
#            it_time = time.time()
            self.opt.step()
#            it_time = time.time() - it_time
            
#            if i % self.log_every == 0:
#                tr_errors.append(err_fn(self.predict(Xtr), Ytr)[0])
#                te_errors.append(err_fn(self.predict(Xte), Yte)[0])
#            it_times.append(it_time)

#        return tr_errors, te_errors, it_times
            
            
#        ft_time = time.time()
            # if i % log_freq == 0:
            #     metrics.append((i, model.compute_metrics(v=opt.model.w, log_test_only=False)))
                
            #     Ktst = ker_fun(Xtst,opt.model.x, False)
            #     preds = Ktst @ opt.model.w
            #     print(Ktst @ opt.model.w)
            #     print(opt.model.K_tst @ opt.model.w)
            #     print(metrics[-1])
            #     print((preds - ytst).square().mean().item() * 0.5)

        
        
        
        
    def predict(self, Xtst):
        K_pred = self.kern_fn(Xtst, self.opt.model.x, False) # I m assuming model is FullKRR
        return K_pred @ self.opt.model.w
    


