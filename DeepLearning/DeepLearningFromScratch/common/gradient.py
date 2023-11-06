import numpy as np

def function_2(x:np.ndarray):
	return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
	"""
	传入函数f, 和需要求偏导的参数x, 输出x的偏导
	"""
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)

	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x1+h)

		x[idx] = tmp_val - h
		fxh2 = f(x) # f(x1-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)

		x[idx] = tmp_val # 还原值
		it.iternext()

	return grad
