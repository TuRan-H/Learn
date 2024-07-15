"""
动手学深度学习
多层感知机的简洁实现
"""
import torch
import matplotlib.pyplot as plt
from torch import nn
from utils.data.mnist import load_data_fashion_mnist
from DiveIntoDeepLearning.chap_4_Multilayer_preceptron.MLP_utils import evaluate_MLP
from tqdm import tqdm




def relu(X):
	a = torch.zeros_like(X)
	return torch.max(X, a)

def net(X):
	X = X.reshape(-1, num_inputs)
	H = relu(X @ W1 + b1)
	return (H @ W2 + b2)




if __name__ == "__main__":
	# 定义超参
	batch_size = 256
	num_epochs = 10
	lr = 0.1

	# 导入数据
	train_loader, test_loader = load_data_fashion_mnist(batch_size)

	# 定义网络, loss函数, 优化器
	num_inputs, num_hiddens, num_outputs = 784, 256, 10
	W1 = nn.Parameter(torch.randn(
		num_inputs, num_hiddens, requires_grad=True) * 0.01)
	b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
	W2 = nn.Parameter(torch.randn(
		num_hiddens, num_outputs, requires_grad=True) * 0.01)
	b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
	params = [W1, b1, W2, b2]

	loss = nn.CrossEntropyLoss(reduction='none')
	updater = torch.optim.SGD(params, lr)

	# train
	for epoch in range(num_epochs):
		bar = tqdm(train_loader, desc=f"training epoch {epoch} / {num_epochs}")
		for x, y in train_loader:
			y_hat = net(x)
			l = loss(y_hat, y)
			updater.zero_grad()
			l.mean().backward()
			updater.step()
			bar.update()

	# evaluate
	evaluate_MLP(net, test_loader, batch_size, show_title=True)