import torch
import torch.nn.functional as F
from mhalib import *
from softmax import FastMaskSoftmax
import copy
from bmm1 import *
from bmm2 import *
from torch_mlu.core.dumptool import dump_cnnl_gencase
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
#dump_cnnl_gencase(enable=True, level='L3')
start = Notifier.Notifier()
end= Notifier.Notifier()
start.place()
mhalib.FastBmm2Dgrad1(mixed, grad_output, grad_mixed, 8, seq, 16, 64, True, True, True, False)
end.place()
end.synchronize()
time = start.hardware_time(end)
print(time)
#dump_cnnl_gencase(enable=False)
print("-------------------")
print(grad_mixed.shape)

print("===check")

g = grad_mixed[0:16*512*512].view(16, 512, 512)

a = mixed[0:512,:].view(512, 16, 3, 64)[:,:,2:3,:]
a = a.squeeze().permute(1,2,0)
print(a.shape)

b = grad_output[0:512]
b = b.squeeze().permute(1,0,2)
print(b.shape)


o = torch.bmm(b, a)
print(o.shape)
print(loss(o, g).cpu())
print(torch.eq(o, g).float().sum().cpu())


g = grad_mixed[16*512*512:16*512*512*2].view(16, 512, 512)

a = mixed[512:1024,:].view(512, 16, 3, 64)[:,:,2:3,:]
a = a.squeeze().permute(1,2,0)
print(a.shape)

b = grad_output[512:1024]
b = b.squeeze().permute(1,0,2)
print(b.shape)


o = torch.bmm(b, a)
print(o.shape)
print(loss(o, g).cpu())
print(torch.eq(o, g).float().sum().cpu())

g = grad_mixed[16*512*512*2:16*512*512*3].view(16, 512, 512)

a = mixed[512*2:512*3,:].view(512, 16, 3, 64)[:,:,2:3,:]
a = a.squeeze().permute(1,2,0)
print(a.shape)

b = grad_output[512*2:512*3]
b = b.squeeze().permute(1,0,2)
print(b.shape)


o = torch.bmm(b, a)
print(o.shape)
print(loss(o, g).cpu())
print(torch.eq(o, g).float().sum().cpu())

