import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import random
from matplotlib import pyplot as plt


class SGD:
	"""
	SGD optimizer

	note
	---
	SGD的优化应该要在反向传播之后, 并且反向传播应使用 l.sum().backward()
	"""
	def __init__(self, params:list, lr, batch_size):
		"""
		params
		---
		params: 所有参数的一个列表, 每个元素都是一个tensor类型
		lr: 学习率
		batch_size: 批量大小
		"""
		self.params = params
		self.lr = lr
		self.batch_size = batch_size
	
	def step(self) -> torch.Tensor:
		with torch.no_grad():
			for param in self.params:
				param -= self.lr * param.grad / self.batch_size
				param.grad.zero_()

class Accumulator:
	"""
	创建一个列表, 列表中的每一项都能够单独计算和
	"""
	def __init__(self, n):
		"""
		params:
		---
		n: 设置Accumulator的data属性一共有多少项
		"""
		self.data = [0.0] * n

	def add(self, *args):
		"""
		向Accumulator中添加数据
		"""
		self.data = [a + float(b) for a, b in zip(self.data, args)]

	def reset(self):
		self.data = [0.0] * len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]


def CrossEntrypyLoss(y_hat:torch.Tensor, y:torch.Tensor):
	"""
	交叉熵损失函数

	params
	---
	y_hat: 模型的预测, shape=(样本的数量, 样本的特征数)
	y: 真实的标签, shape=(样本的数量), 其中y中的每一项都是一个int类型, 表示该样本属于哪一类

	return
	---
	返回一个向量, shape=(样本的数量), 其中每一个元素都代表一个样本的交叉熵的结果
	"""
	# 加上一个1e-9的原因是避免y_hat的值接近0, 如果y_hat接近0的话, 那么loss的结果就趋向于inf
	y_hat = y_hat[list(range(len(y_hat))), y]
	l = -y_hat.log()
	return l.sum()