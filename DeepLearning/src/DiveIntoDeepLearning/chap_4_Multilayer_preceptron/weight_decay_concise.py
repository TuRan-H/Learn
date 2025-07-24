"""
<动手学深度学习>

4.5.2 权重衰减的简介实现
"""
import torch
from dataclasses import dataclass
from torch import nn, optim
from d2l import torch as d2l
from matplotlib import pyplot as plt
from utils.data.synthetic_data import generate_data
from tqdm import tqdm


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


def train(train_loader, test_loader, args:Args):
	# 定义网络, 优化器, 损失函数
	net = nn.Sequential(nn.Linear(args.num_input_features, 1))
	for param in net.parameters():
		nn.init.normal_(param, mean=0, std=1)
	# *** 在这里实现了权值衰减, 对SGD初始化的时候, 会给参数, 对于权重参数, 添加一个 `weight_decay` 字段, 表示权值衰减
	optimizer = optim.SGD(
		[
			{'params':net[0].weight, 'weight_decay':args.lambd},
			{'params':net[0].bias}
		],
		lr = args.lr
	)
	animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='linear',xlim=[5, args.num_epochs], legend=['train', 'test'])
	loss = nn.MSELoss(reduction='mean')

	for epoch in tqdm(range(args.num_epochs)):
		for x, y in train_loader:
			optimizer.zero_grad()
			l = loss(net(x), y)
			l.backward()
			optimizer.step()
		if (epoch + 1) % 5 == 0:
			animator.add(epoch + 1, (d2l.evaluate_loss(net, train_loader, loss), d2l.evaluate_loss(net, test_loader, loss)))
	
	print("L2 norm of w:", net[0].weight.norm().item())
	plt.show()


if __name__ == "__main__":
	args = Args()
	args.num_epochs = 500
	args.lambd = 3
	train_loader, test_loader = generate_data(args)
	train(train_loader, test_loader, args)