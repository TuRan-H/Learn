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