"""
3.3 多维数组的运算
"""
import numpy as np
from matplotlib import pylab as plt


def sigmoid(x:np.array):
	"""
	sigmoid 函数
	:param x:
	"""
	return 1 / (1 + np.exp(-x))

def identity_function(x:np.array):
	"""
	恒等函数
	"""
	return x

def init_network():
	network = {}
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
	network['b1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
	network['b2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
	network['b3'] = np.array([0.1, 0.2])

	return network

def forward(network:dict, x:np.ndarray):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = x.dot(W1) + b1
	z1 = sigmoid(a1)
	a2 = z1.dot(W2) + b2
	z2 = sigmoid(a2)
	a3 = z2.dot(W3) + b3
	z3 = identity_function(a3)

	return z3

x = np.array([1.0, 0.5])
network = init_network()
y = forward(network, x)
print(y)

