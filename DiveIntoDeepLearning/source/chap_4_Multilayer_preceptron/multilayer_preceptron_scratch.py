"""
动手实现 <动手学深度学习> 中的 "多层感知机的从零实现" 
"""
import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
from d2l import torch as d2l
from utils.data import load_fashion_mnist_dataloader
from utils.tools import SGD, get_fashion_mnist_labels, CrossEntrypyLoss


input_size, hidden_size, output_size = 784, 256, 10
w1 = torch.randn(input_size, hidden_size, requires_grad=True)
w2 = torch.randn(hidden_size, output_size, requires_grad=True)
b1 = torch.randn(hidden_size, requires_grad=True)
b2 = torch.randn(output_size, requires_grad=True)
params = [w1, b1, w2, b2]
def net(X:torch.Tensor, batch_size):
	"""
	模型的前向传播
	"""
	try:
		H = X @ w1 + b1
	except:
		X = X.reshape(batch_size, -1)
		H = X @ w1 + b1
	O = relu(H @ w2 + b2)
	return O

def relu(x):
	"""
	relu函数
	"""
	mask = torch.zeros_like(x)
	return torch.max(x, mask)


if __name__ == "__main__":
	batch_size = 256
	epoch_num = 10
	lr = 1e-1
	train_iter, test_iter = load_fashion_mnist_dataloader(batch_size)
	optimizer = SGD(params, lr, batch_size)
	loss = CrossEntrypyLoss

	# train
	for epoch in range(epoch_num):
		for x, y in train_iter:
			x = x.reshape(batch_size, -1)
			y_hat = net(x, batch_size)
			l = loss(y_hat, y)
			l.backward()
			optimizer.step()
			
	# evaluation
	with torch.no_grad():
		x, y = next(iter(test_iter))
		x, y = x[:10], y[:10]
		y_hat = net(x, 10)
		y_hat = torch.argmax(y_hat, dim=1)
		pre_label = get_fashion_mnist_labels(y_hat)
		true_label = get_fashion_mnist_labels(y)
		print(f"{pre_label}\n{true_label}")