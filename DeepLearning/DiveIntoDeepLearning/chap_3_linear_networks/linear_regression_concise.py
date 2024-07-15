"""
动手学深度学习v2
线性回归的简洁实现

模型训练的基本步骤
* 正向传播
* 计算损失
* 置零优化器的梯度
* 反向传播
* 梯度下降
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l


def load_array(data_arrays:tuple, batch_size, is_train=True):
	"""实例化一个DataLoader"""
	dataset = TensorDataset(*data_arrays)
	data_iter = DataLoader(dataset, batch_size, shuffle=is_train)
	return data_iter


class Network(torch.nn.Module):
	def __init__(self, input_features, output_features):
		super().__init__()
		self.net = torch.nn.Linear(input_features, output_features)
		# self.net.weight.data = torch.zeros(self.net.weight.shape, )
		# self.net.bias.data = torch.zeros(self.net.bias.shape)
		self.cal_loss = nn.MSELoss()
	def forward(self, x):
		y = self.net.forward(x)
		return y

	def loss(self, y, y_hat) -> torch.Tensor:
		return self.cal_loss(y, y_hat)



if __name__ == "__main__":
	# 定义网络
	net = Network(2, 1)
	# 定义优化器
	optimizer = torch.optim.SGD(net.parameters(), lr=3*1e-2)
	# 初始化参数
	true_w = torch.tensor([2, -3.4])
	true_b = 4.2

	# 创建数据集
	features, labels = d2l.synthetic_data(true_w, true_b, 1000)
	data_iter = load_array((features, labels), 10)

	# 开始训练
	epoch_num = 3
	for epoch in range(epoch_num):
		for x, y in data_iter:
			prediction = net.forward(x)
			l = net.loss(y, prediction)
			# 置零优化器中的梯度, pytorch会自动进行梯度累加
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
		l = net.loss(net(features), labels)
		print(f"epoch {epoch+1}, loss{l:f}")

	w = net.net.weight.data
	print('estimate error of weight: ', true_w - w.reshape(true_w.shape))
	b = net.net.bias.data
	print('estimate error of bias：', true_b - b)