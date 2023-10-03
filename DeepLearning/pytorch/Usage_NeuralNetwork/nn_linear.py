"""
线性层相关函数
1. flatten函数
flatten函数, 用于将多维的输入展平为n个一维的输出
eg: 输入: torch.size([64, 3, 32, 32])
	输出: torch.size([196608])
	process: 64*3*32*32 = 196608
2. Linear layer: 线性层
https://blog.csdn.net/zhaohongfei_358/article/details/122797190
对输入进行线性变换
"""
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class TuRan(nn.Module):
	def __init__(self, in_feature, out_feature):
		super().__init__()
		# 输入样本的样本为1 * 196608(其中196608=样本的特征数, 样本的batch_size=1), 输出样本的特征数位10
		self.linear = torch.nn.Linear(in_feature, out_feature)

	def forward(self, x):
		y = self.linear(x)
		return y

turan = TuRan(196608, 10)
dataset = CIFAR10("../dataset", False, torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64, drop_last=True)

for datas in dataloader:
	images, targets = datas
	print(images.shape)	# 测试图片的shape为torch.Size([64, 3, 32, 32])
	output_flatten = torch.flatten(images)	# flatten函数, 用于展平化输入
	print(output_flatten.shape)
	output = turan(output_flatten)	# Linear将1*196608的输入线性变化为了1*10的输出
	print(output.shape)