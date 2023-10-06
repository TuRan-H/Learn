import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist



def MSE_Loss_Multiple(output, tag):
	"""
	计算均方误差(batch_size > 1)
	"""
	result = 0.5*np.sum((output-tag)**2)
	return result

def	CrossEntropy_Loss_Multi(output, tag):
	"""
	计算交叉熵误差(batch_size > 1)
	:param output:
	:param tag:
	"""
	delta = 1e-7
	result = tag*np.log(output+delta)
	return -np.sum(result)

def MSE_Loss(y, t):
	"""
	计算均方误差(batch_size >= 1)
	:param y: 神经网络的输出值
	:param t: 标签纸
	:return: 均方误差
	"""
	pass

def CrossEntropy_Loss(y:np.ndarray, t:np.ndarray):
	"""
	计算交叉熵误差(batch_size >= 1)
	:param y: 神经网络的输出
	:param t: 标签值
	:return: 交叉熵误差
	"""
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, t.size)
	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



(x_train, t_train), (x_test, t_test) = load_mnist(True, True, False)
