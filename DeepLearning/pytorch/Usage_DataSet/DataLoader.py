"""
DataLoader的使用
tips:
	1. test_loader可以直接在for循环中调用, test_loader在for循环中使用它的__iter__函数会返回一个iterable
	2. test_loader返回的iterable可以看做
		[data1, data2, data3, data4.....]
		其中
		data1 = (images, targets)
		images = [image1, image2, image3, image4]
		targets = [target1, target2, target3, target4]
		data的数量 = 数据集总个数 / batch_size
		image和target的数量 = batch_size
"""
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision


# 创建一个CIFAR10数据集的实例, 这个实例的每一张图片都经过ToTensor transform
test_data = torchvision.datasets.CIFAR10("../dataset", False, torchvision.transforms.ToTensor())
# 创建一个DataLoader, DataLoader会将CIFAR10实例分割成一个一个batch, batch的大小为64
test_loader = DataLoader(test_data, 64, True, drop_last=False, num_workers=0)


# img, target = test_data[0]
# print(img.shape)
# print(target)
#
# for datas in test_loader:
# 	(images, targets) = datas
# 	print(images.shape)
# 	print(targets)


writer = SummaryWriter("../runs/dataloader")

for epoch in range(2):	# 分成add_image两次, 看看shuffle为true时会不会有影响
	step = 0
	for datas in test_loader:	# test_loader返回值是一个iterable类型
		images, targets = datas	# 对datas解包. datas是一个tuple类型
		writer.add_images(f"test_loader{epoch}", images, step)
		step += 1	# 步进加一
writer.close()