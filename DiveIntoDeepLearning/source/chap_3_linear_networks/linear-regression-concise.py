"""
自己手动实现了书本中 使用了pytorch框架的线性回归网络
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from d2l import torch as d2l


def load_array(data_arrays:tuple, batch_size, is_train=True):
    """实例化一个DataLoader"""
    dataset = data.TensorDataset(*data_arrays)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=is_train)
    return data_iter


# **************************************************主函数部分
net = torch.nn.Sequential(nn.Linear(2, 1))      # 定义模型
loss = nn.MSELoss()                                                      # 定义loss函数
optimizer = torch.optim.SGD(net.parameters(), lr=3*1e-2)          # 定义优化器
true_w = torch.tensor([2, -3.4])                                    # 创建真实的w
true_b = 4.2                                                             # 创建真实的b
features, labels = d2l.synthetic_data(true_w, true_b, 1000)     # 人工创建数据集

data_iter = load_array((features, labels), 10)    # 创建dataloader

epoch_num = 3
for epoch in range(epoch_num):                                           # 迭代部分
    for x, y in data_iter:
        prediction = net(x)
        l = loss(y, prediction)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch+1}, loss{l:f}")

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)