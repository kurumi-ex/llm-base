import torch
from torch import nn
from torch.nn import functional as F
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


class Expert(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, out_features),
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, in_features: int, hidden_dims: int, expert_nums: int, top_k: int, expert_capacity):
        super().__init__()
        self.expert_nums = expert_nums
        self.top_k = top_k
        self.capacity = expert_capacity
        self.out_features = in_features
        self.experts = nn.ModuleList(
            [Expert(in_features, in_features, hidden_dims) for _ in range(expert_nums)]
        )
        self.ffn = nn.Linear(in_features, expert_nums)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        y = self.ffn(x)
        scores = F.softmax(y, dim=-1)
        top_k_probs, top_k_indices = torch.topk(scores, dim=-1, k=self.top_k)

        # 展平
        flat_probs = top_k_probs.view(-1)
        flat_indices = top_k_indices.view(-1)

        # 生成列的索引采样
        sampled_indices = torch.arange(batch_size).reshape(batch_size, 1)

        # 选择最后一维度扩容到 topk
        sampled_indices = sampled_indices.expand(-1, self.top_k).flatten()

        outputs = torch.zeros(batch_size, self.out_features, device=device)

        if self.training:
            importance = scores.sum(dim=0)
            aux_loss = torch.var(importance) / torch.mean(importance)
        else:
            aux_loss = 0

        for idx in range(self.expert_nums):
            expert_mask = idx == flat_indices
            exp_samples = sampled_indices[expert_mask]
            exp_weights = flat_probs[expert_mask]

            if len(exp_samples) == 0:
                # 未选中当前专家
                continue

            # exp_samples是选择的索引列表 将目标的x读入进来
            expert_input = x[exp_samples]

            # 在选择的维度增加一个容量为一的维度
            expert_output = self.experts[idx](expert_input) * exp_weights.unsqueeze(-1)

            # 在选择的维度和索引上进行相加
            outputs.index_add_(0, exp_samples, expert_output)

        return outputs, aux_loss


class GQA(nn.Module):
    def __init__(self, in_features: int,
                 hidden_dims: int,
                 head_nums: int = 8,
                 mask: bool = True,
                 bias: bool = True, ):
        super().__init__()
        self.head_nums = head_nums
        self.hidden_dims = hidden_dims
        self.mask = mask
        self.wq = nn.Linear(in_features, hidden_dims * head_nums, bias=bias)
        self.wk = nn.Linear(in_features, hidden_dims * head_nums // 2, bias=bias)
        self.wv = nn.Linear(in_features, hidden_dims * head_nums // 2, bias=bias)
        self.wo = nn.Linear(hidden_dims * head_nums, in_features, bias=bias)

    def forward(self, x, ):
        batch_size, time_step, vec_size = x.shape
        q = self.wq(x).reshape(batch_size, time_step, self.head_nums, self.hidden_dims)
        k = self.wk(x).reshape(batch_size, time_step, self.head_nums // 2, self.hidden_dims)
        v = self.wv(x).reshape(batch_size, time_step, self.head_nums // 2, self.hidden_dims)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        k = torch.cat([k, k], dim=1)

        scores = F.softmax(q @ k.transpose(-1, -2), dim=-1)

        if self.mask:
            tmp = torch.ones(batch_size, self.head_nums, time_step, time_step, )
            mask = torch.tril(tmp)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            # print(scores)

        v = v.transpose(1, 2)
        v = torch.cat([v, v], dim=1)
        z = scores @ v
        z = z.transpose(1, 2).reshape(batch_size, time_step, self.head_nums * self.hidden_dims)
        return self.wo(z)


if __name__ == "__main__":
    # expert = Expert(in_features=100, out_features=100, hidden_dims=64)
    # myinput = torch.randn(3, 100)
    # print(expert(myinput))
    # x = torch.randn(6, 3)
    # x = F.softmax(x, dim=-1)
    # print(x)
    # _, _x = torch.topk(x, dim=-1, k=2)
    # print(_x)
    # print(_)
    #
    # print(_.view(-1))
    #
    # print(torch.arange(5).reshape(5, 1))
    # tmp = torch.arange(5)[:, None]
    # print(tmp.expand(-1, 2))
    #
    # print(x * torch.tensor([1, 2, 3, 4, 5, 6]).unsqueeze(-1))

    # batch_size = 128
    #
    # x = torch.randn(batch_size, 12, 64, requires_grad=True)
    #
    # model = MoE(
    #     in_features=64,
    #     hidden_dims=256,
    #     expert_nums=10,
    #     top_k=2,
    #     expert_capacity=10,
    # )
    #
    # model.eval()
    #
    # y, _ = model(x.reshape(-1, 64))
    # print(y.shape)
    #
    # x = torch.randn(2, 3)
    # print(x.sum(dim=0))
    # print(torch.mean(x, dim=0))

    model = GQA(in_features=256, hidden_dims=512, )
    input_val = torch.randn(16, 32, 256)
    print(model(input_val).shape)
