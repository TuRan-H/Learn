"""
6.4.1 过拟合
"""

import os, sys
from networkx import network_simplex
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_labmda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay_labmda)
optimizer = SGD()

train_loss_list = []
train_acc_list = []
test_acc_list = []


epoch = 100
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / epoch, 1)
batch_mask = np.random.choice(train_size, batch_size)


epoch_cnt = 0
max_epochs = 201
for i in range(1000000000):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch	= t_train[batch_mask]

	grads = network.gradient(x_batch, t_batch)	# 计算梯度
	optimizer.update(network.params, grads)	# 使用SGD进行梯度下降(随机梯度下降法)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)

		print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

		epoch_cnt += 1
		if epoch_cnt >= max_epochs: break
