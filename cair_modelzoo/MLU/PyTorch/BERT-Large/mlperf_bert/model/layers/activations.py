# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from torch import nn

# Fused GeLU
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456

# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
#@torch.jit.script
def bias_gelu(bias, y):
  x = bias + y
  return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
#@torch.jit.script
def bias_gelu_back(g, bias, y):
  x = bias + y
  tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
  # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
  ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
  return ff*g

class GeLUFunction(torch.autograd.Function):
  @staticmethod
  # bias is an optional argument
  def forward(ctx, input, bias):
    ctx.save_for_backward(input, bias)
    return bias_gelu(bias, input)

  @staticmethod
  def backward(ctx, grad_output):
    input, bias = ctx.saved_tensors
    tmp = bias_gelu_back(grad_output, bias, input)
    return tmp, tmp

bias_gelu_impl = GeLUFunction.apply

# Swish
def swish(x):
  return x * torch.sigmoid(x)

# Fast GeLU
def fast_gelu(x):
  pi = 3.1415926535897932
  cdf = 0.5 * (1.0 + torch.tanh((math.sqrt(2 / pi) * (x + 0.044715 * torch.pow(x, 3)))))
  return x * cdf

def torch_gelu(x):
  return torch.nn.functional.gelu(x)


def torch_bias_gelu(bias, y):
  x = bias + y
  return torch.nn.functional.gelu(x)

ACT2FN = {
  #"gelu": fast_gelu,
  #"bias_gelu": bias_gelu_impl,
  "gelu": torch_gelu,
  "bias_gelu": torch_bias_gelu,
  "relu": torch.nn.functional.relu,
  "swish": swish
}

import torch
import torch_mlu

if __name__ == "__main__":
    bias = torch.randn(4096).to('mlu')
    input = torch.randn([12, 512, 4096]).to('mlu')
    print("===bias_gelu_impl")
    output = bias_gelu_impl(bias, input)
    print("===torch_bias_gelu")
    out_torch = torch_bias_gelu(bias, input)