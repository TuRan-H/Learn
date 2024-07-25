"""
动手学深度学习

4.6 暂退法 (dropout)
"""
import torch
from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from d2l import torch as d2l
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.data.mnist import load_data_fashion_mnist


# ************************************************** substitute tensor __repr__
original_tensor_repr = torch.Tensor.__repr__
def custom_tensor_repr(self):
	return f"Shape:{tuple(self.shape)}, {original_tensor_repr(self)}"
setattr(torch.Tensor, '__repr__', custom_tensor_repr)
# ************************************************** substitute tensor __repr__


@dataclass
class Args:
	dropout_rate1:float = 0.2
	dropout_rate2:float = 0.5
	num_epochs:int = field(default=10)


class Model:
	def __init__(self, args:Args) -> None:
		self.net = nn.Sequential(
			nn.Flatten(),
			nn.Linear(784, 256),
			nn.ReLU(),
			# nn.Dropout(args.dropout_rate1),
			nn.Linear(256, 256),
			nn.ReLU(),
			# nn.Dropout(args.dropout_rate2),
			nn.Linear(256, 10)
		)
		self.net.apply(self.init_weight)

		if torch.cuda.is_available():
			self.net.to('cuda:0')


	def __call__(self, X:torch.Tensor):
		return self.net(X)
	

	@staticmethod
	def init_weight(module:torch.nn.Module):
		"""
		使用正态分布初始化模型线性层的权重
		"""
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, 0, 0.01)
			nn.init.constant_(module.bias, 0)
	
	@property
	def device(self):
		return next(self.net.parameters()).device


def train_epoch(model:Model, train_loader, loss, optimizer):
	# ! 使模型进入训练模式, 这样才能够处理dropout层
	model.net.train()
	metric = d2l.Accumulator(3)

	for x, y in train_loader:
		x, y = x.to(model.device), y.to(model.device)
		y_hat = model(x)
		l = loss(y_hat, y)
		optimizer.zero_grad()
		l.mean().backward()
		optimizer.step()
		metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
	
	return metric[0] / metric[2], metric[1] / metric[2]


def test_epoch(model:Model, test_loader):
	model.net.eval()
	metric = d2l.Accumulator(2)

	with torch.no_grad():
		for x, y in test_loader:
			x, y = x.to(model.device), y.to(model.device)
			metric.add(d2l.accuracy(model(x), y), torch.numel(y))
	
	return metric[0] / metric[1]


def train(args:Args):
	model = Model(args)
	loss = nn.CrossEntropyLoss(reduction='none')
	optimizer = torch.optim.SGD(
		model.net.parameters(), lr=0.5
	)
	animator = d2l.Animator(xlabel='epoch', xlim=[1, args.num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
	train_loader, test_loader = load_data_fashion_mnist(batch_size=256)

	for epoch in tqdm(range(args.num_epochs), desc="Training", disable=True):
		train_loss, train_acc = train_epoch(model, train_loader, loss, optimizer)
		test_acc = test_epoch(model, test_loader)
		animator.add(epoch+1 , [train_loss, train_acc, test_acc])
	plt.show()

		
if __name__ == "__main__":
	args = Args()
	train(args)