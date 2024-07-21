"""
动手学深度学习V2
softmax回归的简洁实现
"""
import os, sys
import torch
import torch.nn as nn

from d2l import torch as d2l
from torch.utils import data
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.data.mnist import load_data_fashion_mnist, FashionMnist_id2label


class SoftmaxNetwork(nn.Module):
	def __init__(self, input_feature, output_feature) -> None:
		super().__init__()
		self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_feature, output_feature))
	
	def forward(self, x:torch.Tensor):
		y = self.net(x)
		return y

def init_weights(layer):
	if isinstance(layer, nn.Linear):
		nn.init.normal_(layer.weight, std=0.01)


def plot_visualization(net:SoftmaxNetwork, dev_dataloader:data.DataLoader):
	x, y = next(iter(dev_dataloader))
	y_hat = net(x)
	y_hat = y_hat.argmax(dim=1)
	labels_predicted = FashionMnist_id2label(y_hat[:10])
	labels = FashionMnist_id2label(y[:10])
	print(labels_predicted)
	print(labels)
	d2l.show_images(x[0:10].reshape([10, 28, 28]), 5, 2, titles=labels_predicted)
	plt.show()



if __name__ == "__main__":
	# 定义超参数
	batch_size = 100
	learning_rate = 0.1
	epoch_num = 10


	# 定义相关实例
	loss = nn.CrossEntropyLoss(reduction="none")
	net = SoftmaxNetwork(784, 10)
	optimizer = torch.optim.SGD(net.parameters(), learning_rate)
	train_loader, test_loader = load_data_fashion_mnist(batch_size)

	for epoch in tqdm(range(epoch_num)):
		for x, y in train_loader:
			net.train()
			y_hat = net(x)
			"""
			* 需要注意的地方
			* 对于loss, 首先loss的参数有先后顺序, 第一个参数是y_hat, 第二个参数是y
			* 其次, loss的参数有dtype要求, 两个参数y_hat和y的dtype都必须是torch.long
			"""
			y.type(torch.long)
			l = loss(y_hat, y)		# 正向传播
			optimizer.zero_grad()	# 清除optimizer的累计梯度
			l.mean().backward()		# 反向传播
			optimizer.step()		# 梯度下降
	
	plot_visualization(net, test_loader)