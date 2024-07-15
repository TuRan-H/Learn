"""
<动手学深度学习>

多层感知机从零开始实现
"""
import torch
import random
from torch import nn
from utils.data.mnist import load_data_fashion_mnist
from tqdm import tqdm
from matplotlib import pyplot as plt
from DiveIntoDeepLearning.chap_4_Multilayer_preceptron.MLP_utils import evaluate_MLP



# 定义网络
class MLP(nn.Module):
	def __init__(self, input_feature, hidden_feature, output_feature):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_feature, hidden_feature, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_feature, output_feature, bias=True)
		)
		# *** 使用net.apply来将init_weight函数按顺序应用到nn.Sequential每一层中
		self.net.apply(self.init_weight)		

	@staticmethod
	def init_weight(m):
		"""
		初始化方法, 随后会应用在net.apply函数中

		Args:
		---
			m: nn.sequential中的某一层
		"""
		if type(m) == nn.Linear:
			nn.init.normal_(m.weight, std=0.01)		# * 使用 nn.init.normal_()函数来正态分布初始化一个网络的weight

	def forward(self, x):
		return self.net(x)




if __name__ == "__main__":
	# 定义超参数
	device = torch.device('cuda:0')
	learning_rate = 1e-1
	batch_size = 256
	num_epoch = 10

	# 实例化对应类
	model = MLP(784, 256, 10)
	model = model.to(device)
	optimizer = torch.optim.SGD(model.parameters(), learning_rate)
	train_loader, test_loader = load_data_fashion_mnist(batch_size, num_workers=4)
	loss = nn.CrossEntropyLoss(reduction="mean")
	train_loss_list = []
	
	# 训练
	model.train()
	for epoch in range(num_epoch):
		bar = tqdm(train_loader, desc=f"training epoch {epoch} / {num_epoch}")
		for x, y in train_loader:
			x = x.reshape(batch_size, -1).to(device)
			y = y.to(device)
			y_hat = model(x)
			l = loss(y_hat, y)
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			train_loss_list.append(l.sum())
			bar.update(1)

	
	# evaluation
	evaluate_MLP(model, test_loader, batch_size, device=device)