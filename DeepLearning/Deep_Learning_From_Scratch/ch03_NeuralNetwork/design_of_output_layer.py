"""
3.5  输出层的设计
"""
import numpy as np

def softmax(x:np.array):
	"""
	实现softmax激活函数, 用来进行分类任务
	"""
	max = np.max(x)
	x = x - max
	exp_a = np.exp(x)
	exp_sum = np.sum(exp_a)

	return exp_a / exp_sum

a = np.array([0.3, 2.9, 4.0])

print(softmax(a))
