import torch
from d2l import torch as d2l


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


def generate_data(args):
	"""
	实现动手学深度学习4.5节 (权重衰减) 中的人工构造数据集
	---
		
	"""
	true_w, true_b = torch.ones([args.num_input_features, 1]) * 0.01, 5
	train_data = d2l.synthetic_data(true_w, true_b, args.num_train)
	train_loader = d2l.load_array(train_data, args.batch_size, True)

	test_data = d2l.synthetic_data(true_w, true_b, args.num_test)
	test_loader = d2l.load_array(test_data, args.batch_size, False)

	return train_loader, test_loader