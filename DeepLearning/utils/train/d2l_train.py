import torch
from torch import nn
from d2l import torch as d2l


def train_convolution_network(model:nn.Module, train_loader, test_loader, num_epochs, learning_rate, weight_decay=None, device=None):
	"""
	"""
	# 参数初始化
	def init_weight(layer):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			nn.init.xavier_uniform_(layer.weight)
	model.apply(init_weight)

	# 定义损失函数, 优化器
	loss = nn.CrossEntropyLoss()
	if weight_decay:
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	else:
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	# 绘图
	num_batches = len(train_loader)
	animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train_loss', 'test_loss', 'test_acc'])

	# 确定模型以及数据的位置
	if not device:
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	model.to(device)

	for epoch in range(num_epochs):
		# metric中, 0表示训练的loss, 1表示测试的loss, 2表示测试的准确率
		metric = d2l.Accumulator(3)
		model.train()
		for i, (X, y) in enumerate(train_loader):
			X, y = X.to(device), y.to(device)
			optimizer.zero_grad()
			y_hat = model(X)
			l = loss(y_hat, y)
			l.backward()
			optimizer.step()
			with torch.no_grad():
				metric.add(l*X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
			train_l = metric[0] / metric[2]
			train_acc = metric[1] / metric[2]
			if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
				animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
		test_acc = d2l.evaluate_accuracy_gpu(model, test_loader)
		animator.add(epoch + 1, (None, None, test_acc))

	print(f"Finally test acc is {test_acc}")	# type: ignore
