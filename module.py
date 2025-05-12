import torch
from torch import nn
import math


class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.2):
        """
        :param in_features:
        :param out_features:
        :param merge: 是否合并
        :param rank: 秩
        :param lora_alpha: 缩放因子 α 对增量进行缩放
        :param dropout:
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout = dropout

        self.fc1 = nn.Linear(in_features, out_features, bias=True)
        self.fc1.weight.requires_grad = False

        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.scale = lora_alpha / rank
            self.initialize_weights()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        if self.rank > 0 and self.merge:
            res = nn.functional.linear(x, self.fc1.weight + self.lora_b @ self.lora_a * self.scale, self.fc1.bias)
            return self.dropout(res)
        return self.dropout(self.fc1(x))

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.lora_a.data, a=math.sqrt(5))
        print(self.lora_a.data)
        nn.init.zeros_(self.lora_b.data)
