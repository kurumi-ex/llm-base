import numpy as np
import torch
import torch.nn as nn


def softmax(x: np.ndarray):
    """
    assume x[B T V]
    """
    # 减去最大值防止特大的值
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    exp_sum = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / exp_sum


def cross_entropy(prediction: np.ndarray, labels: np.ndarray):
    """
    prediction[batch_size class_nums]
    labels[batch_size]
    :param prediction:
    :param labels:
    :return:
    """
    softmax_x = softmax(prediction)
    probs = softmax_x[np.arange(len(labels)), labels]
    n = labels.shape[0]
    return np.sum(-np.log(probs)) / n


# x = np.array([
#     [1, 2, 3],
#     [4, 0, 6],
# ], dtype=np.float32)
# x = x.reshape((1, 2, 3))
# print(x)
# print(softmax(x))

logits = np.array([
    [2.0, 1.0, 0.1, -1.2],  # 第一个样本的logits
    [1.2, 3.1, 0.0, -0.5],  # 第二个样本的logits
    [0.0, -1.0, 2.5, 0.3]  # 第三个样本的logits
])

# 真实标签（对应每个样本的正确分类索引）
targets = np.array([0, 1, 2])

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(torch.tensor(logits), torch.tensor(targets))

print(f"my result: {cross_entropy(logits, targets)}")
print(f"torch result: {loss.item()}")

print(np.max(logits, axis=-1))
