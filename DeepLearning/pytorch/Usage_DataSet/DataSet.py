"""
1. 如何将DataSet用在TorchVision中
2. 如何在DataSet的实例化时运用transform
3. DataSet的调用以及返回值
"""
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

trans_ToTensor = transforms.ToTensor()	# 创建变化, 供给后面的数据集使用

set_train = datasets.CIFAR10(root="../dataset", train=True, transform=trans_ToTensor)	# 实例化数据集, 所有数据都会通过trans_ToTensor进行一次变化
set_test = datasets.CIFAR10(root="../dataset", train=False, transform=trans_ToTensor)	# train的true和false表示选定的数据集到底是测试集还是训练集


writer = SummaryWriter("../runs/exp2")
for i in range(10):
	image, target = set_test[i]		# 数据集是可调用的, __getitem__函数返回的是一个元组(tuple) 第一个是image(图像), 第二个是target(标签)
	writer.add_image("set_test", image, i)
writer.close()