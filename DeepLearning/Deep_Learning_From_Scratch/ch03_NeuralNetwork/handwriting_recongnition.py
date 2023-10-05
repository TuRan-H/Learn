"""
利用Mnist数据集和给定的已训练好的network来实现手写数字识别
"""
import pickle
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
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

def sigmoid(x:np.array):
	"""
	sigmoid 函数
	:param x:
	"""
	return 1 / (1 + np.exp(-x))

def img_show(img):
	"""
	输出图片
	"""
	pil_image = Image.fromarray(np.uint8(img))
	pil_image.show()

def get_data():
	"""
	获取Mnist测试数据集
	"""
	(x_train, t_train), (x_test, t_test) = load_mnist(True, True ,False)
	return x_test, t_test

def init_network():
	"""
	从sample_weight.pkl文件中获取神经网络的参数
	"""
	with open("sample_weight.pkl", "rb") as fp:
		network = pickle.load(fp)
	return network

def predict(network:dict, x:np.ndarray) -> np.ndarray:
	"""
	将输入的x通过神经网络走一遍并输出y
	"""
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)
	return y

def get_accuary(network, input, tag, batch_size):
	"""
	获取 精确率
	"""
	accuracy_cnt = 0
	for i in range(0, len(input), batch_size):
		x_batch = input[i:i+batch_size]
		y_batch = predict(network, x_batch)
		p = np.argmax(y_batch, 1)
		for j in range(100):
			if p[j] == tag[i + j]:
				accuracy_cnt += 1
	accuracy_rate = accuracy_cnt / len(input)
	return accuracy_rate


x, t = get_data()
network = init_network()
# accuracy_rate = get_accuary(network, x, t, 100)
# for i in range(10):
# 	img_show(x[i])
