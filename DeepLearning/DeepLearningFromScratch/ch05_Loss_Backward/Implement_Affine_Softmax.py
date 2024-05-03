"""
5.6 Affine/Softmax层的实现
	5.6.1 Affine层
	5.6.2 批版本的Affine层
	5.6.3 Softmax-with-loss层的实现
"""
import numpy as np
from common.functions import *

class Affine:
	"""
	Affine层的正向传播反向传播
	"""
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		x_dot_w = np.dot(x, self.W)
		output = x_dot_w + self.b

		return output

	def backward(self, d_out):
		dx = np.dot(d_out, self.W.T)
		self.dW = np.dot(self.x.T, d_out)
		self.b = np.sum(d_out, axis=0)

		return dx

class SoftMaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, d_out=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size

		return dx