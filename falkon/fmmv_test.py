import torch
import falkon
from pykeops.torch import Genred

A = torch.randn(1000, 10)
B = torch.randn(1000, 10)
v = torch.randn(1000, 1)

opt = falkon.FalkonOptions(keops_active="force", use_cpu=False, debug=True)
kernel = falkon.kernels.GaussianKernel(1.2)
out = kernel.mmv(A, B, v, opt=opt)

print(out)
