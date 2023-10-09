"""
4.5  学习算法的实现
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist


class TwoLayer:
	def __init__(self,input_size, hidden_size, output_size, weight_init_std=0.01):
		"""
		@param input_size: 输入层的神经元数目
		@param hidden_size: 隐藏层的神经元数目
		@param output_size: 输出层的神经元数目
		@param weight_init_std: 标准权重初始值
		"""
		self.param = {}
		# 创建字典, 字典中存入的值都是神经网络的参数
		self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.param['b1'] = np.zeros_like(hidden_size)
		self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.param['b2'] = np.zeros_like(output_size)

	def predict(self, x):
		"""
		神经网络处理x
		@param x: 神经网络的输入值
		@return: 神经网络的输出值
		"""
		W1, W2 = self.param['W1'], self.param['W2']
		b1, b2 = self.param['b1'], self.param['b2']
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y


	def loss(self, x, t):
		"""
		计算神经网络处理之后的输出值和标签之间的loss
		@param x: 神经网络的输入值
		@param t: 标签值
		@return: loss
		"""
		y = self.predict(x)
		return cross_entropy_error(y, t)

	def accuracy(self, x, t):
		"""
		计算精确度
		@param x: 神经网络的输入值
		@param t: 标签值
		@return:精确度
		"""
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)

		return np.sum(y==t) / float(x.shape[0])

	def numercal_gradient(self, x, t):
		"""
		loss(x, t)函数对权重和偏置求梯度
		@param x: 神经网络的输入值
		@param t: 标签值
		@return: 梯度的字典
		"""
		loss_W = lambda W: self.loss(x, t)
		grad = {}	# 定义一个gradient字典
		grad['W1'] = numerical_gradient(loss_W, self.param['W1'])
		grad['W2'] = numerical_gradient(loss_W, self.param['W2'])
		grad['b1'] = numerical_gradient(loss_W, self.param['b1'])
		grad['b2'] = numerical_gradient(loss_W, self.param['b2'])

		return grad


net = TwoLayer(784, 50, 10)
(x_train, t_train), (x_test, t_test) = load_mnist()

# ***定义超参数***
batch_size = 100
learning_rate = 0.1
iters_num = 10
# ***定义基础变量***
train_size = x_train.shape[0]
train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
iter_per_epoch = max(train_size / batch_size, 1)


for i in range(iters_num):
	batch_mask = np.random.choice(batch_size, train_size)	# 获取batch
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	grad = net.numercal_gradient(x_batch, t_batch)	# 计算梯度

	for key in ('W1', 'W2', 'b1', 'b2'):
		net.param[key] = grad[key]

	loss = net.loss(x_batch, t_batch)
	train_loss_list.append(loss)
	if i % iter_per_epoch == 0:
		train_accuracy = net.accuracy(x_train, t_train)
		test_accuracy = net.accuracy(x_test, t_test)
		train_accuracy_list.append(train_accuracy)
		test_accuracy_list.append(test_accuracy)
	print(f"{train_loss_list}\n{train_accuracy_list}\n{test_accuracy_list}")

