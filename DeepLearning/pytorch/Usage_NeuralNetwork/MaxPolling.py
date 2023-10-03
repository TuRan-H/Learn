"""
池化层(polling layer)的使用案例
https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class TuRan(nn.Module):
	def __init__(self):
		super().__init__()
		self.maxpoll = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

	def forward(self, x):
		y = self.maxpoll(x)
		return y

dataset = CIFAR10(root="../dataset", train=False, transform=ToTensor())
dataloader = DataLoader(dataset, 64, False)

# 创建输入, 供给后来池化使用
input = torch.tensor(([1, 2, 0, 3, 1],
					  [0, 1, 2, 3, 1],
					  [1, 2, 1, 0, 0],
					  [5, 2, 3, 1, 1],
					  [2, 1, 0, 1, 1]),
					 dtype=torch.float32
					 )
input = torch.reshape(input, (-1, 1, 5, 5))	# shape元组的含义: (batch size, channel数, 高, 宽)
# 输出input, 查看数据
print(input)

turan = TuRan()
# 进行池化兵输出
output_tensor = turan(input)
# 查看池化输出
print(output_tensor)

writer = SummaryWriter("../runs/exp4")

step = 0
for datas in dataloader:
	images, targets = datas
	output = turan(images)	# 将每一张图片都进行池化
	writer.add_images("original", images, step)
	writer.add_images("MaxPolling",output, step)
	step += 1

writer.close()