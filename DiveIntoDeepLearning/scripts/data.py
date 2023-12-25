import torch
import torchvision
from torchvision import transforms
from torch.utils import data

def load_data_fashion_mnist(batch_size, resize:torch.Size=None):
	trans = [transforms.ToTensor()]
	if resize:
		trans.insert(0, transforms.Resize(resize))
	minist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans, download=True)
	minist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans, download=True)
	return (data.DataLoader(minist_train, batch_size, num_workers=4, shuffle=True), 
			data.DataLoader(minist_test, batch_size, num_workers=True, shuffle=False))

if __name__=="__main__":
	load_data_fashion_mnist(10)