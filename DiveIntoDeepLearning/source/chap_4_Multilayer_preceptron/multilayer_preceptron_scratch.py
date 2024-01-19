"""
动手实现 <动手学深度学习> 中的 "多层感知机的从零实现" 
"""
import torch
from torch import nn
from d2l import torch as d2l
from utils.tools import get_fashion_mnist_labels, evaluate_MLP, show_images
import matplotlib.pyplot as plt


# 获取数据集, 定义超参
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_epochs = 10
lr = 0.1


# 定义网络, loss函数, 优化器
num_inputs, num_hiddens, num_outputs = 784, 256, 10
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

def relu(X):
	a = torch.zeros_like(X)
	return torch.max(X, a)

def net(X):
	X = X.reshape(-1, num_inputs)
	H = relu(X @ W1 + b1)
	return (H @ W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')
updater = torch.optim.SGD(params, lr)


# train
for epoch in range(num_epochs):
	for x, y in train_iter:
		y_hat = net(x)
		l = loss(y_hat, y)
		updater.zero_grad()
		l.mean().backward()
		updater.step()


# evaluate
evaluate_MLP(net, test_iter, batch_size, show_title=True)
plt.show()