"""
如何利用convolution 2D创建那一个自己的神经网络
"""
from torch import nn, reshape
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


dataset = CIFAR10("../dataset", False, ToTensor())
dataloader = DataLoader(dataset, 64, shuffle=False, drop_last=True)

class TuRan(nn.Module):
	"""
	创建一个自己的模型 TuRan
	模型继承与nn.Module
	"""
	def __init__(self):
		# super函数使得子类可以使用父类的方法, 并且将子类的self传递到父类替代父类的self引用, super的参数可以省略
		super(TuRan, self).__init__()
		# 创建一个Conv2d对象, 输入通道为3, 输出通道为6, 卷积核大小为3*3矩阵, 步进为1, 填充为0
		self.conv1 = Conv2d(3, 6, (3, 3), 1, 0)

	def forward(self, x):
		"""
		定义一个forward函数, 使实例可以被直接调用
		:param x:
		:return: 卷积后的x
		"""
		x = self.conv1(x)
		return x

turan = TuRan()
writer = SummaryWriter("../runs/exp3")

step = 0
for datas in dataloader:
	images, targets = datas
	writer.add_images("original", images)
	step += 1

step = 0
for datas in dataloader:
	images, targets = datas
	output = turan(images)	# 对图像进行卷积
	# 由于reshape要求图像的channel只能为3或者4, 因此必须将其reshape, 设置channel为4, 大小为30*30, -1代表让函数自己计算batch size
	output = reshape(output, (-1, 3, 30,30))
	writer.add_images("convolution", output, step)
	step += 1

writer.close()