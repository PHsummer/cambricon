import torch
import torch.nn.functional as F
from mhalib import *
from softmax import FastMaskSoftmax
import copy
from bmm1 import *
from bmm2 import *
import torch_mlu.core.device.notifier as Notifier


torch.manual_seed(66)

InitMHACUDAExtension()

loss = torch.nn.MSELoss()
bmm=Bmm2Strided(None, None, 16, 64)
seq=torch.tensor([512, 512, 512, 512, 512, 512, 512, 512]).int()
print(bmm)

grad_output=torch.randn(4096, 16, 64).to('mlu').half()
mixed=torch.randn(4096, 3072).to('mlu').half()
grad_mixed=torch.randn(512*512*16*8).to('mlu').half()

print("-------------------")
start = Notifier.Notifier()
end= Notifier.Notifier()
start.place()
mhalib.FastBmm2Dgrad2(grad_output, grad_mixed, mixed, 8, seq, 16, 64, True, True, True, False)
end.place()
end.synchronize()
time = start.hardware_time(end)
print(time)
print("-------------------")
print(grad_mixed.shape)

print("===check")

g = grad_mixed[0:16*512*512].view(16, 512, 512)
g = g.permute(0,1,2)
print(g.shape)
#a = mixed[0:512,:].view(512, 16, 3, 64)[:,:,2:3,:]
#a = a.squeeze().permute(1,2,0)
#print(a.shape)

b = grad_output[0:512]
b = b.squeeze().permute(1,2,0)
print(b.shape)


c = mixed[0:512,:].view(512, 16, 3, 64)[:,:,2:3,:]
c = c.squeeze().permute(1, 2, 0)

print("----------check")
o = torch.bmm(b, g)
print(loss(o, c).cpu())
print(torch.eq(o, c).float().sum().cpu())


g = grad_mixed[16*512*512*1:16*512*512*2].view(16, 512, 512)
g = g.permute(0,1,2)
print(g.shape)
#a = mixed[0:512,:].view(512, 16, 3, 64)[:,:,2:3,:]
#a = a.squeeze().permute(1,2,0)
#print(a.shape)

b = grad_output[512*1:512*2]
b = b.squeeze().permute(1,2,0)
print(b.shape)


c = mixed[512*1:512*2,:].view(512, 16, 3, 64)[:,:,2:3,:]
c = c.squeeze().permute(1, 2, 0)

print("----------check")
o = torch.bmm(b, g)
print(loss(o, c).cpu())
print(torch.eq(o, c).float().sum().cpu())


g = grad_mixed[16*512*512*2:16*512*512*3].view(16, 512, 512)
g = g.permute(0,1,2)
print(g.shape)

b = grad_output[512*2:512*3]
b = b.squeeze().permute(1,2,0)
print(b.shape)


c = mixed[512*2:512*3,:].view(512, 16, 3, 64)[:,:,2:3,:]
c = c.squeeze().permute(1, 2, 0)

print("----------check")
o = torch.bmm(b, g)
print(loss(o, c).cpu())
print(torch.eq(o, c).float().sum().cpu())


