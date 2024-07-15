"""
动手学深度学习V2
线性回归的从零实现
"""
import random
import torch

def synthetic_data(w:torch.Tensor, b:torch.Tensor, num_examples:int):
	"""
	人工创建数据集

	params
	---
	w: 权重
	b: 偏置
	num_examples: 样本的数量

	return
	---
	返回一个元组, 分别是 (人工构建的输入, 根据输入计算出来的真实的输出)
	公式为 `y = x^T W + b + \epsilon`
	"""
	X = torch.normal(0, 1, (num_examples, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape((-1, 1))


class data_iter:
	def __init__(self, batch_size, features, labels):
		"""
		batch_size: 一个batch中样本数的大小
		features: 模型的输入数据, 是一个矩阵. shape是(样本的个数, 一个样本的特征数)
		labels: 真实的标签, shape是(样本的个数, 1)
		"""
		self.features = features
		self.labels = labels
		self.batch_size = batch_size
		self.num_examples = len(features)
		self.indices = list(range(self.num_examples))
		# 创建一个列表, 这个列表总的每一项都是一个tensor, 这个tensor代表了一个batch中每个元素的indices
		self.batch_indices = []		
		self.count = -1
		random.shuffle(self.indices)
		for i in range(0, self.num_examples, self.batch_size):
			batch_indice = torch.tensor(self.indices[i:min(i + self.batch_size, self.num_examples)])
			self.batch_indices.append(batch_indice)

	def __next__(self) -> tuple(torch.Tensor, torch.Tensor):
		self.count += 1
		try: 
			return self.features[self.batch_indices[self.count]], self.labels[self.batch_indices[self.count]]
		except IndexError:
			print("the hole of data has been iterated, reset the data_loader")
			self.count = -1
			raise StopIteration

	def __iter__(self):
		return self


class Model:
	def __init__(self, w:torch.Tensor, b:torch.Tensor):
		"""
		w: 模型的权重
		b: 模型的偏置
		"""
		self.w = w
		self.b = b
	
	def forward(self, x):
		"""
		正向传播

		params
		---
		x: 模型的输入

		return
		---
		模型的输入x经过模型正向传播后的输出
		"""
		return torch.matmul(x, self.w) + self.b
	
	def __call__(self, x):
		return self.forward(x)


class MSE:
	def __init__(self):
		self.mean_square_error = lambda y_hat, y: (y_hat - y) ** 2 / 2
	
	
	def __call__(self, y_hat, y) -> torch.Tensor:
		return self.mean_square_error(y_hat, y)


class SGD:
	def __init__(self, params:torch.Tensor, lr, batch_size):
		self.params = params
		self.lr = lr
		self.batch_size = batch_size
	
	def update(self) -> torch.Tensor:
		with torch.no_grad():
			for param in self.params:
				"""
				* 由于之前计算loss的时候进行了 l.sum()
				* \frac{\partial l.sum()}{\partial w}  = \sum_{i}^{n}{(x_iw^T-y)x_i^T} 
				* 其中, x_i表示输入矩阵X的第几行(输入矩阵中的每一行代表一个单独的输入)
				* 如果在数学上直接对l求解梯度, 那么得出来的w.grad应该是一个矩阵, 
				* 但是对l.sum()求解梯度就对这个矩阵进行sum(axis=1)
				"""
				param -= self.lr * param.grad / self.batch_size
				param.grad.zero_()

	


if __name__ == "__main__":
	true_w = torch.tensor([2, -3.4])
	true_b = 4.2
	num_examples = 1000
	features, labels = synthetic_data(true_w, true_b, num_examples)
	data_loader = data_iter(10, features, labels)

	# 初始化权重
	w = torch.zeros((2, 1), requires_grad=True, dtype=torch.float)
	b = torch.zeros((1), requires_grad=True, dtype=torch.float)

	# 定义超参数和相关实例
	batch_size = 10
	lr = 1e-2
	num_epochs = 10
	net = Model(w, b)
	loss = MSE()
	optimizer = SGD([w, b], lr, batch_size)

	# 开始训练
	for epoch in range(num_epochs):
		for x, y in data_loader:
			prediction = net(x)
			l = loss(y, prediction)
			"""
			* 这里将l求和, 对求和的结果进行反向传播
			* 这样做的原因是: 如果不对l求和l就不是一个标量, pytorch中不可以对非标量进行反向传播
			* 这里对l求和, 将l压缩成了一个标量, 就可以对其进行反向传播
			"""
			l.sum().backward()
			optimizer.update()
		with torch.no_grad():
			train_l = loss(net(features), labels)
			print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
	
	print(w.shape)
			
	
	# 查看真实的w和模型预测出来的w之间差距大小
	print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
	print(f'b的估计误差: {true_b - b}')