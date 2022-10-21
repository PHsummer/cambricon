import torch
import torch.nn.functional as F
from mhalib import *
from softmax import FastMaskSoftmax
import copy
import torch_mlu
from bmm1 import *

import torch_mlu.core.device.notifier as Notifier
torch.set_printoptions(profile="full")

torch.manual_seed(66)

InitMHACUDAExtension()

loss = torch.nn.MSELoss()
bmm=Bmm1Strided(None, None, 16, 64)
seq=torch.tensor([512, 512, 512, 512, 512, 512, 512, 512]).int()
print(bmm)

input=torch.randn(4096, 3072).to('mlu').half()

start = Notifier.Notifier()
end= Notifier.Notifier()
start.place()
print("-------------------")
#output = bmm(input, 8, seq)
output = bmm(input, 8, seq)
print("-------------------")
end.place()
end.synchronize()
time = start.hardware_time(end)
print(time)

o0 = input[0:512].view(512, 16, 3, 64).permute([2, 1, 0, 3])[0]
o1 = input[0:512].view(512, 16, 3, 64).permute([2, 1, 0, 3])[1]


base=torch.bmm(0.125 * o1, o0.permute(0,2,1)).permute([0,2,1]).reshape(-1)
#base=torch.bmm(o1, o0.permute(0,2,1)).permute([0,2,1]).reshape(-1)
print(base.cpu().view(16, 512, 512)[0,0:5,0:5])
print(output[0][:16*512*512].view(16, 512, 512)[0,0:5,0:5])

print(loss(base, output[0][:16*512*512]).cpu())
print(torch.eq(base.cpu().float(), output[0][:16*512*512].cpu().float()).float().sum().cpu())

o0 = input[512:1024].view(512, 16, 3, 64).permute([2, 1, 0, 3])[0]
o1 = input[512:1024].view(512, 16, 3, 64).permute([2, 1, 0, 3])[1]

base=torch.bmm(0.125 * o1, o0.permute(0,2,1)).permute([0,2,1]).reshape(-1)
#print(base.cpu())
#print(output[0][16*512*512:16*512*512*2].cpu())
print(base.cpu().view(16, 512, 512)[0,0:5,0:5])
print(output[0][:16*512*512].view(16, 512, 512)[0,0:5,0:5])
print(loss(base, output[0][16*512*512:16*512*512*2]).cpu())
print(torch.eq(base, output[0][16*512*512:16*512*512*2]).float().sum().cpu())

o0 = input[1024:1536].view(512, 16, 3, 64).permute([2, 1, 0, 3])[0]
o1 = input[1024:1536].view(512, 16, 3, 64).permute([2, 1, 0, 3])[1]

base=torch.bmm(0.125 * o1, o0.permute(0,2,1)).permute([0,2,1]).reshape(-1)
#print(base.cpu())
#print(output[0][16*512*512*2:16*512*512*3].cpu())
print(base.cpu().view(16, 512, 512)[0,0:5,0:5])
print(output[0][:16*512*512].view(16, 512, 512)[0,0:5,0:5])
print(loss(base, output[0][16*512*512*2:16*512*512*3]).cpu())
print(torch.eq(base, output[0][16*512*512*2:16*512*512*3]).float().sum().cpu())


