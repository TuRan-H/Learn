"""
动手学深度学习

章节6.6 LeNet
"""

import torch
from torch import nn
from d2l import torch as d2l
from typing import Iterable
from tqdm import tqdm
from matplotlib import pyplot as plt


# class Reshape(nn.Module):
# 	def __init__(self, *args, **kwargs) -> None:
# 		super().__init__(*args, **kwargs)
	

# 	def forward(self, X):
# 		return X.reshape()


class LeNet(nn.Module):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.model = nn.Sequential(
			nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
			nn.AvgPool2d(2, 2),
			nn.Conv2d(6, 16, 5), nn.Sigmoid(),
			nn.AvgPool2d(2, 2),
			nn.Flatten(),
			nn.Linear(16*5*5, 120), nn.Sigmoid(),
			nn.Linear(120, 84), nn.Sigmoid(),
			nn.Linear(84, 10)
		)
	

	def forward(self, X):
		for i in self.model:
			X = i(X)
		return X


def evaluate_accuracy_gpu(net:nn.Module, data_iter:Iterable, device=None):
	"""
	评估模型在指定设备上的准确率

	Args:
		net: 神经网络
		data_iter: 测试集数据迭代器
	"""
	net.eval()
	if device == None:
		device = next(iter(net.parameters())).device
	
	metric = d2l.Accumulator(2)
	with torch.no_grad():
		for X, y in data_iter:
			X, y = X.to(device), y.to(device)
			metric.add(d2l.accuracy(net(X), y), y.numel())
	
	return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
	# 初始化参数
	def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d:
			nn.init.xavier_uniform_(m.weight)
	net.apply(init_weights)
	net.to(device)

	# 定义优化器和损失函数
	optimizer = torch.optim.SGD(net.parameters(), lr=lr)
	loss = nn.CrossEntropyLoss()
	animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
	timer, num_batches = d2l.Timer(), len(train_iter)

	for epoch in tqdm(range(num_epochs)):
		# 训练损失之和，训练准确率之和，样本数
		metric = d2l.Accumulator(3)
		net.train()

		# 迭代测试集
		for i, (X, y) in enumerate(train_iter):
			timer.start()
			optimizer.zero_grad()
			X, y = X.to(device), y.to(device)
			y_hat = net(X)
			l = loss(y_hat, y)
			l.backward()
			optimizer.step()
			with torch.no_grad():
				metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
			timer.stop()
			train_l = metric[0] / metric[2]
			train_acc = metric[1] / metric[2]
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
		
		# 训练完成, 进行评估
		test_acc = evaluate_accuracy_gpu(net, test_iter)
		animator.add(epoch + 1, (None, None, test_acc))

	print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
	print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ', f'on {str(device)}')


if __name__ == '__main__':
	# 定义超参数
	net = LeNet()
	batch_size = 256
	train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
	lr, num_epochs = 0.9, 10
	train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
	plt.show()