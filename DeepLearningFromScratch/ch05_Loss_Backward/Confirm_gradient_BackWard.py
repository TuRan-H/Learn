"""
5.7.3 误差反响传播法的梯度确认
"""
import sys,os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common import functions,model

(image_train, tag_train), (image_test, tag_test) = load_mnist(True, False, True)
two_layer = model.TwoLayer(784, 50, 10)
x_batch = image_test[:3]
t_batch = tag_test[:3]
num_grad = two_layer.numerical_gradient(x_batch, t_batch)
grad = two_layer.gradient(x_batch, t_batch)

for key in grad.keys():
	diff = np.average(np.abs(num_grad[key] - grad[key]))
	print(key +":" + str(diff))