"""
5.4 简单层的实现
	5.4.1 乘法层的实现
	5.4.2 加法层的实现
"""
import numpy as np

class MulLayer:
	"""
	简单神经网络中的乘法层计算正向传播和反向传播
	"""
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self, x, y):
		"""
		计算正向传播
		@return: 正向传播的输出值
		"""
		self.x = x
		self.y = y
		output = self.x * self.y
		return output

	def backward(self, d_out):
		"""
		计算反向传播
		@param d_out: 下一层传过来的导数
		@return: 上一层的导数
		"""
		d_x = d_out * self.y
		d_y = d_out * self.x
		return d_x, d_y

class AddLayer:
	def	__init__(self):
		pass

	def forward(self, x, y):
		# 计算正向传播
		output = x + y
		return output

	def backward(self, d_out):
		# 计算反向传播
		d_x = d_out
		d_y = d_out
		return d_x, d_y

mul = MulLayer()
output = mul.forward(1, 2)
d_output_x, d_output_y = mul.backward(1)
print(f"{output}, {d_output_x}, {d_output_y}")


add = AddLayer()
output = add.forward(1, 2)
d_output_x, d_output_y = add.backward(1)
print(f"{output}, {d_output_x}, {d_output_y}")
