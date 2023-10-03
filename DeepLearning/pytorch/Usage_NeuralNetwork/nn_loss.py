"""
1. 如何通过loss function计算输出可目标之间的差距
	1. L1LOSS
	Creates a criterion that measures the mean absolute error (MAE) between each element in the input
 target y.
	2. MSELOSS
	Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input
x and target y.
	3. CROSSENTROPYLOSS
2. 为后序更新输出提供反向传播(backward)
3. 梯度(grad)下降的概念
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Conv2d, MaxPool2d, Flatten, Linear
# from torch.nn import


class TuRan(torch.nn.Module):
	"""创建测试用的神经网络"""
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

# 创建数据集和dataloader
dataset = torchvision.datasets.CIFAR10("../dataset", False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, 1)


# L1LOSS 测试
input_L1LOSS = torch.tensor([1, 2, 3], dtype=torch.float32)
target_L1LOSS = torch.tensor([1, 2, 4], dtype=torch.float32)
loss_L1LOSS = L1Loss()
output_L1Loss = loss_L1LOSS(input_L1LOSS, target_L1LOSS)
# print(output_L1Loss)

# MSELOSS 测试
input_MESLOSS = torch.tensor([1, 2, 3], dtype=torch.float32)
target_MESLOSS = torch.tensor([1, 2, 5], dtype=torch.float32)
loss_MESLOSS = MSELoss()
output_MSELOSS = loss_MESLOSS(input_MESLOSS, target_MESLOSS)
# print(output_MSELOSS)

# CROSSENTROPYLOSS 测试
input_CROSSENTROPYLOSS = torch.tensor([0.1, 0.2, 0.3])
input_CROSSENTROPYLOSS = torch.reshape(input_CROSSENTROPYLOSS, [1, 3])	# 将输入修改为函数指定的shape(N, C)
target_CROSSENTROPYLOSS = torch.tensor([1])	# target的shape是(C), 表示target指代的是哪一个class
loss_CROSSENTROPYLOSS = CrossEntropyLoss()
result_CROSSENTROPYLOSS = loss_CROSSENTROPYLOSS(input_CROSSENTROPYLOSS, target_CROSSENTROPYLOSS)

# 对CIFAR10数据集进行loss(CrossEntropyLoss)测试
turan = TuRan()
loss = CrossEntropyLoss()
for datas in dataloader:
	images, targets = datas
	output = turan(images)	# 经过网络进行变化
	result = loss(output, targets)	# 计算输出和target之间的差距
	result.backward() # 对结果进行反向传播
