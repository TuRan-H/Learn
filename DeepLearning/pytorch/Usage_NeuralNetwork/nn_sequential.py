"""
1. 利用add_graph查看网络模型的具体结构
2. 利用torch.ones创建测试数据集
3. 利用sequential搭建模型
4. 尝试学习公式, 并且利用已知的数据求解公式中的其他数据, 在合适的位置填写参数

"""
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class TuRan(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = Conv2d(3, 32, 5, 1, 2)
		self.maxpool1 = MaxPool2d(2, 2, 0)
		self.conv2 = Conv2d(32, 32, 5, 1, 2)
		self.maxpool2 = MaxPool2d(2, 2, 0)
		self.conv3 = Conv2d(32, 64, 5, 1, 2)
		self.maxpool3 = MaxPool2d(2, 2, 0)
		self.flatten = Flatten()
		self.linear1 = Linear(1024, 64)
		self.linear2 = Linear(64, 10)
		self.seq = torch.nn.Sequential(
			self.conv1,
			self.maxpool1,
			self.conv2,
			self.maxpool2,
			self.conv3,
			self.maxpool3,
			self.flatten,
			self.linear1,
			self.linear2
		)

	def forward(self, x):
		y = self.seq(x)
		return y

turan = TuRan()
writer = SummaryWriter("../runs/sequential")

input = torch.ones([64, 3, 32, 32])	# 创建测试数据集, 数据集的所有元素全部都是1
output = turan(input)

writer.add_graph(turan, input)	# 利用add_graph来可视化变换过程
writer.close()