import torch
from d2l import torch as d2l

def Generate_data(args):
	"""
	生成人工数据集
	---
		
	"""
	true_w, true_b = torch.ones([args.num_input_features, 1]) * 0.01, 5
	train_data = d2l.synthetic_data(true_w, true_b, args.num_train)
	train_loader = d2l.load_array(train_data, args.batch_size, True)

	test_data = d2l.synthetic_data(true_w, true_b, args.num_test)
	test_loader = d2l.load_array(test_data, args.batch_size, False)

	return train_loader, test_loader