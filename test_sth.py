import torch
from torch import nn
from module import *

lora = LoraLinear(32, 128, True)
x = torch.randn(64, 32, requires_grad=False)
print(lora(x))
