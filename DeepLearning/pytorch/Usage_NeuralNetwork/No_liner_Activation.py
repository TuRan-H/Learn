"""
非线性激活的两个函数
1. Relu函数
ReLU(非线性变换)的DOCS
https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
2. Sigmoid函数
Sigmoid函数DOCS
https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
"""
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10


class TuRan(nn.Module):
	def __init__(self):
		super().__init__()
		self.sigmoid = torch.nn.Sigmoid()
		self.relu = torch.nn.ReLU(False)

	def forward(self, input):
		output = self.sigmoid(input)
		return output

	def get_relu(self, input):
		output = self.relu(input)
		return output

turan = TuRan()

input = torch.tensor([[1, -9],
					  [-1, 3]])
print(f"input:\n{input}\n")
output = turan(input)	# 利用sigmoid处理input
print(f"sigmoid output:\n{output}\n")
output = turan.get_relu(input)	# 利用relu处理input
print(f"relu output:\n{output}\n")


# 创建数据集, 查看relu和sigmoid对于图像的处理
dataset = CIFAR10("../dataset", False, torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 64, False)
writer = SummaryWriter("../runs/exp5")

step = 0
for datas in dataloader:
	(images, targets) = datas
	writer.add_images("original", images, step)
	output = turan(images)
	writer.add_images("Sigmoid", output, step)
	output = turan.get_relu(images)
	writer.add_images("ReLU", output, step)
	step += 1
writer.close()