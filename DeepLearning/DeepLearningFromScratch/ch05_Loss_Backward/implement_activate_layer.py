"""
5.5 激活函数层的实现
	5.5.1 relu函数的实现
"""
import numpy as np

class Relu:
	def	__init__(self):
		self.mask = None

	def forward(self, x:np.ndarray):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, d_out):
		d_out[self.mask] = 0

		return d_out

class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		output = 1 / (1 + np.exp(-x))
		self.out = output

		return output

	def backward(self, d_out):
		d_x = d_out * self.out * (1 - self.out)

		return d_x

print("hello world")