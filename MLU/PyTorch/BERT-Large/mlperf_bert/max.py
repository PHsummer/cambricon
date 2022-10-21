import torch
import torch.nn.functional as F
import torch_mlu
from mhalib import *
from softmax import FastMaskSoftmax
import copy

torch.manual_seed(66)

import torch_mlu.core.device.notifier as Notifier

InitMHACUDAExtension()
m1 = FastMaskSoftmax(dim=-1)
input=torch.randn(11714560).to('mlu').half()
input.requires_grad=True
c_input = copy.deepcopy(input)
mask=torch.randn(2272).to('mlu').half()
seq=torch.tensor([448, 368, 336, 320, 288, 240, 176, 96])
print("-----------------")
start = Notifier.Notifier()
end= Notifier.Notifier()
start.place()
out = m1(input, mask, 8, seq, 16)
end.place()
out.backward(out)
end.synchronize()
time = start.hardware_time(end)
print(time)
print("-----------------")

#print("-------------------------")
#count=0
#mc=0
#for i in range(seq.shape[0]):
#    s = seq[i].item()
#    n = 16
#
#    #print(torch.equal(F.softmax(c_input[count: count + n*s*s].view(n*s, s) + mask[mc:mc+s], dim=-1).view(-1), out[count: count + n * s * s]))
#    print("%d:%d" % (count, count + n * s * s))
#    print(torch.equal(F.softmax(c_input[count: count + n*s*s].view(n*s, s) + mask[mc:mc+s], dim=-1).view(-1), out[count: count + n * s * s]))
#    count += (n * s * s)
#    mc += (s)

