"""
实现一个简单的神经网络
实现计算数值微分(梯度)
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3)	# W: 神经网络的权重

	def predict(self, x):
		"""
		神经网络的输入值和权重相乘之后的值
		@param x: 神经网络的输入值
		@return: 神经网络的输出值
		"""
		result = np.dot(x, self.W)
		return result

	def loss(self, x, t):
		"""
		计算神经网络的输入值和标签之间的loss
		@param x: 神经网络的输入值
		@param t: 标签值
		@return: loss值
		"""
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y, t)

		return loss



net = SimpleNet()	# 创建实例
x = np.array([0.6, 0.9])	# 创建输入x
p = net.predict(x)	# x通过神经网络的输出
t = np.zeros_like(p)	# 根据输出创建标签
t[p.argmax()] = 1

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)