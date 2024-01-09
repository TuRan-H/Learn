import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import random
from matplotlib import pyplot as plt


class SGD:
	"""
	SGD optimizer

	note
	---
	SGD的优化应该要在反向传播之后, 并且反向传播应使用 l.sum().backward()
	"""
	def __init__(self, params:list, lr, batch_size):
		"""
		params
		---
		params: 所有参数的一个列表, 每个元素都是一个tensor类型
		lr: 学习率
		batch_size: 批量大小
		"""
		self.params = params
		self.lr = lr
		self.batch_size = batch_size
	
	def step(self) -> torch.Tensor:
		with torch.no_grad():
			for param in self.params:
				param -= self.lr * param.grad / self.batch_size
				param.grad.zero_()

def get_fashion_mnist_labels(labels):
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

def CrossEntrypyLoss(y_hat:torch.Tensor, y:torch.Tensor):
	"""
	交叉熵损失函数

	params
	---
	y_hat: 模型的预测, shape=(样本的数量, 样本的特征数)
	y: 真实的标签, shape=(样本的数量), 其中y中的每一项都是一个int类型, 表示该样本属于哪一类

	return
	---
	返回一个向量, shape=(样本的数量), 其中每一个元素都代表一个样本的交叉熵的结果
	"""
	# 加上一个1e-9的原因是避免y_hat的值接近0, 如果y_hat接近0的话, 那么loss的结果就趋向于inf
	y_hat = y_hat[list(range(len(y_hat))), y]
	l = -y_hat.log()
	return l.sum()

def evaluate_MLP(net, test_iter, batch_size, device=None):
	"""
	评估MLP的准确性

	params
	---
	net: 神经网络模型, 需要有__call__方法
	test_iter: 测试数据集的迭代器
	batch_size: 批量大小
	device: 使用的device
	"""
	index = [random.randint(0, batch_size) for _ in range(10)]
	with torch.no_grad():
		x, y = next(iter(test_iter))
		x, y = x[index], y[index]
		x_origin = x
		x = x.reshape(10, -1)
		if device:
			x = x.to(device)
			y = y.to(device)
			net = net.to(device)
		y_hat = net(x)
		y_hat = torch.argmax(y_hat, dim=1)
		show_images(x_origin, 2, 5, titles=y, titles_pred=y_hat)

def show_images(imgs, num_rows, num_cols, titles=None, titles_pred=None, scale=2):
	"""
	显示一系列的图像

	params
	---
	imgs: matplotlib支持的图像数据, shape=(H, W, C)
	num_rows: subplot的行数
	num_clos: subplot的列数
	titles: 真实的标签
	titles_pred: 预测出来的标签
	scale: figure的大小
	"""
	figsize = (num_cols*scale, num_rows*scale)
	_, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
	plt.subplots_adjust(wspace=0.5, hspace=0.5)		# 调整子图之间的距离
	axes = axes.flatten()
	if titles is not None:  # ! 一个多值的张量不能使用 if title来判定是否为空
		titles = get_fashion_mnist_labels(titles)
	if titles_pred is not None:
		titles_pred = get_fashion_mnist_labels(titles_pred)
	for i, (ax, img) in enumerate(zip(axes, imgs)):
		try:
			ax.imshow(img)
		except:
			img = img.numpy()
			# ! matplotlib接受的图像shape应该是(H, W, C), 而fashion取出来是(C, H, W)
			img = img.transpose(1, 2, 0)
			ax.imshow(img)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.set_title(titles[i])
		if titles_pred:
			ax.set_title(f"{titles[i]}\n{titles_pred[i]}")
	return axes