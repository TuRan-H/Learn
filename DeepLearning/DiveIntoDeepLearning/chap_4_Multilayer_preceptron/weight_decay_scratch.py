"""
<动手学深度学习>

4.5.2 权重衰减的从零实现
"""
import torch

from dataclasses import dataclass
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
from utils.data.synthetic_data import Generate_data


@dataclass
class Args:
	"""
	Class to store the arguments for the experiment.

	Attributes:
		num_train (int): Number of training examples.
		num_test (int): Number of testing examples.
		num_input_features (int): Number of input features.
		batch_size (int): Batch size for training.
		num_epochs (int): Number of training epochs.
		lr (float): Learning rate for the optimizer.
		lambd (float): 权值衰减率
	"""
	num_train: int = 20
	num_test: int = 100
	num_input_features: int = 200
	batch_size: int = 5
	num_epochs: int = 100
	lr: float = 1e-3
	lambd: float = 0


def l2_penalty(w:torch.Tensor):
	"""
	实现权重衰减, L2范数惩罚.

	$||W||$
	"""
	return torch.sum(w.pow(2)) / 2


def init_params(w_shape=[200, 1]):
	w = torch.normal(0, 1, w_shape, requires_grad=True)
	b = torch.zeros(1, requires_grad=True)

	return w, b


def train(train_loader, test_loader, args:Args):
	"""
	"""
	# 初始化
	w, b =init_params()
	net = lambda X: torch.matmul(X, w) + b
	loss = lambda y_hat, y: (y_hat - torch.reshape(y, y_hat.shape))**2 / 2
	animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, args.num_epochs], legend=['train', 'test'])

	# 开始训练
	for epoch in range(args.num_epochs):
		for x, y in train_loader:
			l = loss(net(x), y) + args.lambd * l2_penalty(w)
			l.sum().backward()
			d2l.sgd([w, b], args.lr, args.batch_size)
		if (epoch + 1) % 5 == 0:
			animator.add(epoch + 1, (d2l.evaluate_loss(net, train_loader, loss), d2l.evaluate_loss(net, test_loader, loss)))

	print("hello world")
	plt.show()
	print("L2 norm of w:", torch.norm(w).item())




if __name__ == "__main__":
	args = Args()
	args.lambd = 3

	train_loader, test_loader = Generate_data(args)
	train(train_loader, test_loader, args)