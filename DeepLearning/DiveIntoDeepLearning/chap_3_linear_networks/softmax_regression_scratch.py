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
from utils.data.mnist import load_data_fashion_mnist


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


class SGD:
	def __init__(self, params:torch.Tensor, lr, batch_size):
		self.params = params
		self.lr = lr
		self.batch_size = batch_size
	
	def update(self) -> torch.Tensor:
		with torch.no_grad():
			for param in self.params:
				"""
				* 由于之前计算loss的时候进行了 l.sum()
				* \frac{\partial l.sum()}{\partial w}  = \sum_{i}^{n}{(x_iw^T-y)x_i^T} 
				* 其中, x_i表示输入矩阵X的第几行(输入矩阵中的每一行代表一个单独的输入)
				* 如果在数学上直接对l求解梯度, 那么得出来的w.grad应该是一个矩阵, 
				* 但是对l.sum()求解梯度就对这个矩阵进行sum(axis=1)
				"""
				param -= self.lr * param.grad / self.batch_size
				param.grad.zero_()


def CrossEntrypyLoss(y_hat:torch.Tensor, y:torch.Tensor):
	"""
	交叉熵损失函数

	params
	---
	y_hat: 模型的预测, shape=(样本的数量, 样本的特征数)
	y: 真实的标签, shape=(样本的数量), 其中y中的每一项都是一个int类型, 表示该样本属于哪一类

	return
	---
	返回一个张量, 这个张量是一个向量求sum的结果 (向量, shape=(样本的数量), 其中每一个元素都代表一个样本的交叉熵的结果)
	"""
	y_hat = y_hat[list(range(len(y_hat))), y]
	return -y_hat.log()


def id2label(labels):
	"""
	根据给出的labels, 获取真实的labels

	note
	---
	输入labels是一个列表, 列表中的每一个元素都是int类型, 表示一个类别号
	比如说labels[0] = 1, 那么就说明第0号样本的标签为trouser

	params
	---
	labels: 使用数字表示的标签列表
	"""
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
				   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	
	return [text_labels[int(i)] for i in labels]



def plot_visualization(net:SoftmaxNetwork, dev_dataloader:data.DataLoader):
	x, y = next(iter(dev_dataloader))
	y_hat = net(x)
	y_hat = y_hat.argmax(dim=1)
	labels_predicted = id2label(y_hat)
	labels = id2label(y)
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
			optimizer.update()
		for x, y in test_dataloader:
			with torch.no_grad():
				y_hat = net(x)
				l = loss(y_hat, y)
				loss_list.append(l.sum())

	print(loss_list[::16])
	plot_visualization(net, dev_dataloader)
