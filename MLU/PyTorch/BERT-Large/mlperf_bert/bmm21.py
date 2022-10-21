import torch
import torch.nn.functional as F
from mhalib import *
from softmax import FastMaskSoftmax
import copy
from bmm1 import *
from bmm2 import *


torch.manual_seed(66)

InitMHACUDAExtension()

loss = torch.nn.MSELoss()
bmm=Bmm2Strided(None, None, 16, 64)
seq=torch.tensor([512, 512, 512, 512, 512, 512, 512, 512]).int()
print(bmm)

grad_output=torch.randn(512*512*8*16).to('mlu').half()
mixed=torch.randn(4096, 3072).to('mlu').half()
grad_mixed=torch.randn(4096, 16, 64).to('mlu').half()

print("-------------------")
mhalib.FastBmm2Fprop(mixed, grad_output, grad_mixed, 8, seq, 16, 64, True, True, True, False)
print("-------------------")
print(grad_mixed.shape)

print("===check")

g = grad_mixed[0:512,:].view(512, 16, 64)
g = g.squeeze().permute(1,2,0)
print(g.shape)

a = mixed[0:512,:].view(512, 16, 3, 64)[:,:,2:3,:]
a = a.squeeze().permute(1,2,0)
print(a.shape)

b = grad_output[0:16*512*512].view(16, 512, 512).permute([0,2,1])
print(b.shape)


o = torch.bmm(a, b)
print(o.shape)
print(loss(o, g).cpu())
print(torch.eq(o, g).float().sum().cpu())


g = grad_mixed[512:1024,:].view(512, 16, 64)
g = g.squeeze().permute(1,2,0)
print(g.shape)

a = mixed[512:1024,:].view(512, 16, 3, 64)[:,:,2:3,:]
a = a.squeeze().permute(1,2,0)
print(a.shape)

#b = grad_output[16*512*512*1:16*512*512*2].view(16, 512, 512).permute([0,2,1])
b = grad_output[16*512*512*1:16*512*512*2].view(16, 512, 512).permute([0,2,1])
print(b.shape)

o = torch.bmm(a, b)
print(o.shape)
print(loss(o, g).cpu())
print(torch.eq(o, g).float().sum().cpu())

