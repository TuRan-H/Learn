"""
误差反向传播法的实现
"""
import sys, os
sys.path.append(os.pardir)
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
		self.layers['Affien1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affien2'] = Affine(self.params['W2'], self.params['b2'])
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

	def accuracy(self, x, t):
		"""
		计算神经网络输出的精确度
		@param x: 神经网络的输入
		@param t: 神经网络的标签
		@return:
		"""
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		if t.ndim != 1: t = np.argmax(t, axis=1)
		total = np.sum(y==t)

		return total / float(t.shape[0])

	def numerical_gradient(self, x, t):
		pass

	def gradient(self, x, t):
		pass

	def get_gradient(self):
		pass