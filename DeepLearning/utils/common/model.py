import sys, os
sys.path.append(os.getcwd())
from collections import OrderedDict
from common.functions import *
import numpy as np
from common.gradient import numerical_gradient


class TwoLayer:
	def	__init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		# 初始化权重
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['b2'] = np.zeros(output_size)

		# 生成层
		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.last_layer = SoftmaxWithLoss()

	def predict(self, x):
		"""
		x通过神经网络的输出
		@param x: 神经网络的输入
		@return: 神经网络的输出
		"""
		for layer in self.layers.values():
			x = layer.forward(x)

		return x

	def loss(self, x, t):
		y = self.predict(x)
		loss = self.last_layer.forward(y, t)

		return loss

	def accuracy(self, x, t):
		"""
		计算神经网络输出的精确度
		@param x: 神经网络的输入
		@param t: 神经网络的标签
		@return:
		"""
		y = self.predict(x)		# 预测神经网络的输出值
		y = np.argmax(y, axis=1)		# 输出值是一个(nxn)的矩阵, 通过argmax函数以列为轴取最大值, 将最大值的index构建成一个新的矩阵(1xn)
		if t.ndim != 1: t = np.argmax(t, axis=1)
		accuracy = np.sum(y==t) / float(t.shape[0])

		return accuracy

	def gradient(self, x, t, dout=1):
		grads = {}
		loss = self.loss(x, t)
		dout = self.last_layer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)

		grads['W1'] = self.layers['Affine1'].dW
		grads['W2'] = self.layers['Affine2'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['b2'] = self.layers['Affine2'].db

		return grads

	def numerical_gradient(self, x, t):
		"""
		求对于loss函数所有层的偏导

		:param x: 神经网络的输入
		:param t: 标签值
		"""
		loss_W = lambda W:self.loss(x, t)
		grads = {}

		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads