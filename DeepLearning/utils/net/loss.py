import torch


class MSE:
	def __init__(self):
		self.mean_square_error = lambda y_hat, y: (y_hat - y) ** 2 / 2
	
	
	def __call__(self, y_hat, y) -> torch.Tensor:
		return self.mean_square_error(y_hat, y)


def CrossEntrypyLoss(y_hat:torch.Tensor, y:torch.Tensor):
	"""
	交叉熵损失函数

	params
	---
	y_hat: 模型的预测, shape=(样本的数量, 样本的特征数)
	y: 真实的标签, shape=(样本的数量), 其中y中的每一项都是一个int类型, 表示该样本属于哪一类

	return
	---
	返回一个向量, shape=(样本的数量), 其中每一个元素都代表一个样本的交叉熵的结果
	"""
	# 加上一个1e-9的原因是避免y_hat的值接近0, 如果y_hat接近0的话, 那么loss的结果就趋向于inf
	y_hat = y_hat[list(range(len(y_hat))), y]
	l = -y_hat.log()
	return l.sum()