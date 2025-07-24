"""
动手学深度学习V2
softmax回归的从零实现
"""
import torch
import torchvision

from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.data.mnist import load_data_fashion_mnist, FashionMnist_id2label
from utils.net.loss import CrossEntrypyLoss
from utils.net.SGD import SGD



d2l.use_svg_display()




class SoftmaxNetwork:
	def __init__(self, features, cls):
		"""
		params
		---
		features: 输入向量有多少个feature
		cls: 最终分类的时候有多少个class
		"""
		self.weight = torch.normal(0, 1, (features, cls), requires_grad=True)
		self.bias = torch.zeros(cls, requires_grad=True)


	def affine(self, x:torch.Tensor):
		"""
		全连接层, 置于softmax层之前

		params
		---
		x: 输入矩阵, shape为(样本数, 特征数)

		return
		---
		全连接层的输出, size是 (样本数量, 分类类别数)
		"""
		x = x.reshape(-1, self.weight.shape[0])
		# torch.matmul() 矩阵惩罚, 
		output = torch.matmul(x, self.weight)
		output += self.bias
		
		return output
	

	def softmax(self, x:torch.Tensor):
		"""
		softmax层, 接收的输入是 全连接层的输出

		params
		---
		x: 全连接层的输出, shape为 (样本数, 分类类别数)

		return
		---
		softmax层的输出
		"""
		x_exp = torch.exp(x)
		partition = torch.sum(x_exp, dim=1, keepdim=True)

		return x_exp / partition
	

	def __call__(self, x:torch.Tensor):
		y = self.affine(x)
		y = self.softmax(y)

		return y









def plot_visualization(net:SoftmaxNetwork, dev_dataloader:data.DataLoader):
	x, y = next(iter(dev_dataloader))
	y_hat = net(x)
	y_hat = y_hat.argmax(dim=1)
	labels_predicted = FashionMnist_id2label(y_hat)
	labels = FashionMnist_id2label(y)
	d2l.show_images(x[0:10].reshape([10, 28, 28]), 5, 2, titles=labels_predicted)
	plt.show()

	


if __name__ == "__main__":
	# 定义超参数
	num_epoch = 10
	learn_rate = 1e-1
	input_features = 784
	output_features = 10
	batch_size = 256

	# 定义对应的实例
	train_dataloader, test_dataloader = load_data_fashion_mnist(batch_size)
	dev_dataloader = test_dataloader
	net = SoftmaxNetwork( input_features,output_features)
	optimizer = SGD((net.weight, net.bias), learn_rate, batch_size)
	loss = lambda y_hat, y: CrossEntrypyLoss(y_hat, y)
	loss_list = []

	# 开始训练
	for epoch in tqdm(range(num_epoch)):
		for x, y in train_dataloader:
			y_hat = net(x)
			l = loss(y_hat, y)
			l.sum().backward()
			optimizer.step()
		for x, y in test_dataloader:
			with torch.no_grad():
				y_hat = net(x)
				l = loss(y_hat, y)
				loss_list.append(l.sum())

	print(loss_list[::16])
	plot_visualization(net, dev_dataloader)
