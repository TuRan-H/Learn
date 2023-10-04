"""
3.2 激活函数
"""
import numpy as np
import matplotlib.pylab as plt


def step_function(x:np.ndarray):
	"""
	实现阶跃函数, 输入一个nparray类型的x, 将其转化为阶跃函数的输出
	:param x: np.ndarray
	"""
	return (x>0).astype(int)

def sigmoid(x:np.array):
	"""
	sigmoid 函数
	:param x:
	"""
	return 1 / (1 + np.exp(-x))

def ReLU(x:np.array):
	"""
	rectify linear unit 函数
	:param x:
	"""
	return 	np.maximum(x, 0)

# 创建一个numpy数组, 用于测试
x = np.arange(-5.0, 5.0, 0.1)
y = ReLU(x)

# 画图
plt.plot(x, y, linestyle='-')
plt.ylim(-0.1, 1.1)
plt.show()