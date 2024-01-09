"""
动手实现:  <动手学深度学习> 多层感知机的简洁实现
"""
# %%
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.data import load_fashion_mnist_dataloader
from utils.tools import get_fashion_mnist_labels, evaluate_MLP


# %%
# 定义网络
class MLP(nn.Module):
	def __init__(self, input_feature, hidden_feature, output_feature):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_feature, hidden_feature, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_feature, output_feature, bias=True)
		)
		self.net.apply(self.init_weight)		# ! 使用net.apply来将init_weight函数按顺序应用到nn.Sequential每一层中

	@staticmethod
	def init_weight(m):
		"""
		初始化方法, 随后会应用在net.apply函数中
		! 网络的初始化一定不能够忘记

		params
		---
		m: nn.sequential中的某一层
		"""
		if type(m) == nn.Linear:
			nn.init.normal_(m.weight, std=0.01)		# * 使用 nn.init.normal_()函数来正态分布初始化一个网络的weight

	def forward(self, x):
		return self.net(x)


if __name__ == "__main__":
	# 定义超参数和相关实例
	device = torch.device('cuda:0')
	learning_rate = 1e-1
	batch_size = 256
	num_epoch = 10
	net = MLP(784, 256, 10)		# 定义网络
	net = net.to(device)
	optimizer = torch.optim.SGD(net.parameters(), learning_rate)		# 定义优化器
	train_iter, test_iter = load_fashion_mnist_dataloader(batch_size, num_workers=4)		# 获取训练集和测试集
	loss = nn.CrossEntropyLoss(reduction="mean")		# 定义loss函数
	train_loss_list = []
	
	# train
	net.train()
	for epoch in range(num_epoch):
		for x, y in train_iter:
			x = x.reshape(batch_size, -1).to(device)
			y = y.to(device)
			y_hat = net(x)
			l = loss(y_hat, y)
			optimizer.zero_grad()	# ! 一定要加上optimizer.zero_grad() 将梯度重置一下, 要不然后面的梯度会进行累积
			l.backward()
			optimizer.step()
			train_loss_list.append(l.sum())
	
	
	# %%
	# evaluation
	evaluate_MLP(net, test_iter, batch_size, device=device)