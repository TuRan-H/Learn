"""
5.7.4  使用误差反向传播法的学习
"""
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.model import TwoLayer
from dataset.mnist import load_mnist


model = TwoLayer(784, 50, 10)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
iter_num = 10000		# 迭代次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size/batch_size, 1)


epoch = 0
for i in range(iter_num):
	mask = np.random.choice(train_size, batch_size)
	x_train_batch = x_train[mask]
	t_train_batch = t_train[mask]

	# 求解梯度
	grad = model.gradient(x_train_batch, t_train_batch)

	# 参数更新
	for key in ['W1', 'W2', 'b1', 'b2']:
		model.params[key] -= learning_rate * grad[key]

	loss = model.loss(x_train_batch, t_train_batch)
	train_loss_list.append(loss)

	# 每个epoch记录一下并且输出
	if i % iter_per_epoch == 0:
		train_acc = model.accuracy(x_train, t_train)
		test_acc = model.accuracy(x_test, t_test)
		test_acc_list.append(test_acc)
		train_acc_list.append(train_acc)
		print(f"iter step {i} in {epoch} epoch\n the test accuracy is {test_acc}\nthe train accuracy is {train_acc} ")
		epoch += 1
