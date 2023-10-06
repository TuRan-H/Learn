import numpy as np


def MSE_Loss_single(output, tag):
	"""
	计算均方误差
	"""
	result = 0.5*np.sum((output-tag)**2)
	return result

def	CrossEntropy_Loss_single(output, tag):
	"""
	计算交叉熵误差
	:param output:
	:param tag:
	"""
	delta = 1e-7
	result = tag*np.log(output+delta)
	return -np.sum(result)

