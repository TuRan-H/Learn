import sys, os
sys.path.append(os.pardir)
import numpy as np

def sigmoid(x):
	"""
	sigmoid 激活函数
	"""
	return 1 / (1 + np.exp(-x))

def softmax(x):
	"""
	softmax 输出层函数
	"""
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T

	x = x - np.max(x) # 溢出对策
	return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
	"""
	交叉熵 loss函数
	@param y: 神经网络的输出
	@param t: 标签
	"""
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	# 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
	if t.size == y.size:
		t = t.argmax(axis=1)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Affine:
	def __init__(self, W, b):
		self.W =W
		self.b = b

		self.x = None
		self.original_x_shape = None
		# 权重和偏置参数的导数
		self.dW = None
		self.db = None

	def forward(self, x):
		# 对应张量
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x

		out = np.dot(self.x, self.W) + self.b

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)

		dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
		return dx


class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None # softmax的输出
		self.t = None # 监督数据

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size

		return dx


class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx
