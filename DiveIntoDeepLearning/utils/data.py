import torch
import torchvision
from torchvision import transforms
from torch.utils import data

def load_fashion_mnist_dataloader(batch_size, resize:torch.Size=None, num_workers=4):
	"""
	获取Fashion Mnist数据集的train datalaoder和test dataloader

	params
	---
	batch_size: 批量大小
	resize: 修改Fashion mnist中每一张图片的size, 默认值为None

	return
	---
	(train_dataloader, test_dataloader)
	"""

	trans = [transforms.ToTensor()]
	if resize:
		trans.insert(0, transforms.Resize(resize))
	trans = transforms.Compose(trans)
	minist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans, download=True)
	minist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans, download=True)
	return (data.DataLoader(minist_train, batch_size, num_workers=num_workers, shuffle=True, drop_last=True), 
			data.DataLoader(minist_test, batch_size, num_workers=num_workers, shuffle=False, drop_last=True))

def load_fashion_mnist_dev_dataloader():
	"""
	获取验证集
	"""
	dev_set = torchvision.datasets.FashionMNIST(root="data", train=False, transform=transforms.Compose([transforms.ToTensor()]), download=False)
	return data.DataLoader(dev_set, 10, shuffle=False, num_workers=4)