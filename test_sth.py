import torch
from torch import nn
from module import *

# lora = LoraLinear(32, 128, True)
# x = torch.randn(64, 32, requires_grad=False)
# print(lora(x))


import numpy as np

a=np.random.rand(2,2,2)
print(a)
b=np.random.rand(2,2,2)
print(b)
print(np.dot(a,b))
