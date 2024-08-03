"""
动手学深度学习

4.10 实战Kaggle比赛：预测房价
"""
import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
from sympy import linear_eq_to_matrix
import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR


# 定义数据集下载连接
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
	DATA_URL + 'kaggle_house_pred_train.csv',
	'585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
	DATA_URL + 'kaggle_house_pred_test.csv',
	'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


class Model(nn.Module):
	"""
	定义模型
	"""
	def __init__(self, in_features):
		super(Model, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(in_features, 64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(64, 16),
			nn.ReLU(),
			nn.Linear(16, 1)
		)

	def forward(self, X):
		return self.net(X)


def download(name, cache_dir=os.path.join('.', 'dataset')):  #@save
	"""下载一个DATA_HUB中的文件，返回本地文件名"""
	assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
	url, sha1_hash = DATA_HUB[name]
	os.makedirs(cache_dir, exist_ok=True)
	fname = os.path.join(cache_dir, url.split('/')[-1])
	if os.path.exists(fname):
		sha1 = hashlib.sha1()
		with open(fname, 'rb') as f:
			while True:
				data = f.read(1048576)
				if not data:
					break
				sha1.update(data)
		if sha1.hexdigest() == sha1_hash:
			return fname  # 命中缓存
	print(f'正在从{url}下载{fname}...')
	r = requests.get(url, stream=True, verify=True)
	with open(fname, 'wb') as f:
		f.write(r.content)
	return fname


def download_extract(name, folder=None):  #@save
	"""下载并解压zip/tar文件"""
	fname = download(name)
	base_dir = os.path.dirname(fname)
	data_dir, ext = os.path.splitext(fname)
	if ext == '.zip':
		fp = zipfile.ZipFile(fname, 'r')
	elif ext in ('.tar', '.gz'):
		fp = tarfile.open(fname, 'r')
	else:
		assert False, '只有zip/tar文件可以被解压缩'
	fp.extractall(base_dir)
	return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  #@save
	"""下载DATA_HUB中的所有文件"""
	for name in DATA_HUB:
		download(name)


def log_rmse(net, features, labels):
	# 为了在取对数时进一步稳定该值，将小于1的值设置为1, torch.clamp是裁剪函数, 避免取对数时出现负无穷
	clipped_preds = torch.clamp(net(features), 1, float('inf'))
	# 这里采用的是log形式的mse, 因为房价在计算的时候具有相对性, 因此这里计算相对差距
	rmse = torch.sqrt(loss(torch.log(clipped_preds),
						   torch.log(labels)))

	return rmse.item()


def train(
	net, 
	train_features,
	train_labels,
	test_features,
	test_labels,
	num_epochs,
	learning_rate,
	weight_decay,
	batch_size
):
	"""
	训练函数
	"""
	# train_ls, test_ls 用来记录训练时训练集和测试集的loss
	train_ls, test_ls = [], []
	train_iter = d2l.load_array((train_features, train_labels), batch_size)
	# 使用adam优化器
	optimizer = torch.optim.Adam(
		net.parameters(),
		lr = learning_rate,
		weight_decay = weight_decay
	)

	bar = tqdm(range(num_epochs), desc="training epoch: 1")
	for epoch in range(num_epochs):
		for X, y in train_iter:
			optimizer.zero_grad()
			l = loss(net(X), y)
			l.backward()
			optimizer.step()
		train_ls.append(log_rmse(net, train_features, train_labels))
		if test_labels is not None:
			test_ls.append(log_rmse(net, test_features, test_labels))
		bar.update()
		bar.set_description(f"training epoch: {epoch + 1}")
	return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
	"""
	获取k则交叉验证的训练集和验证集

	Args:
		k: int, 折数
		i: int, 当前是第几折
		X: tensor, 所有的特征
		y: tensor, 所有的标签

	Returns:
		X_train, y_train, X_valid, y_valid: 训练集特征，训练集标签，验证集特征，验证集标签
	"""
	assert k > 1
	fold_size = X.shape[0] // k
	X_train, y_train = None, None
	for j in range(k):
		# slice用于创建一个切片对象, 能够作为索引使用
		idx = slice(j * fold_size, (j + 1) * fold_size)
		X_part, y_part = X[idx, :], y[idx]
		if j == i:
			# 如果i == j, 则把当前的x_part, y_part作为验证集
			X_valid, y_valid = X_part, y_part
		elif X_train is None:
			X_train, y_train = X_part, y_part
		else:
			X_train = torch.cat([X_train, X_part], 0)
			y_train = torch.cat([y_train, y_part], 0)	#type: ignore

	return X_train, y_train, X_valid, y_valid	# type: ignore


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size, save_dir=None):
	"""
	k折交叉验证
	"""
	train_l_sum, valid_l_sum = 0, 0
	for i in range(k):
		data = get_k_fold_data(k, i, X_train, y_train)
		net = Model(in_features)
		train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
								   weight_decay, batch_size)
		train_l_sum += train_ls[-1]
		valid_l_sum += valid_ls[-1]
		if i == 0:
			d2l.plot(
				list(range(1, num_epochs + 1)),
				[train_ls, valid_ls],
				xlabel='epoch',
				ylabel='rmse',
				xlim=[1, num_epochs],
				legend=['train', 'valid'],
				yscale='log'
			)
			
			if save_dir:
				plt.savefig(os.path.join(save_dir, 'k_fold_average.svg'))
			else:
				plt.show()


		print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
			  f'验证log rmse{float(valid_ls[-1]):f}')

	return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size, save_dir=None):
	net = Model(in_features)
	train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
	d2l.plot(
		np.arange(1, num_epochs + 1), 
		[train_ls], 
		xlabel='epoch', 
		ylabel='log rmse', 
		xlim=[1, num_epochs], 
		yscale='log'
	)
	
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'train.svg'))
	else:
		plt.show()

	print(f'训练log rmse：{float(train_ls[-1]):f}')

	# 将网络应用于测试集。
	preds = net(test_features).detach().numpy()
	# 将其重新格式化以导出到Kaggle
	test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
	submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
	if save_dir != None:
		submission.to_csv(os.path.join(save_dir, "submission.csv"), index=False)


if __name__ == "__main__":
	# 定义超参
	k = 5
	num_epochs = 100
	lr = 3e-2
	weight_decay = 0
	batch_size = 64
	save_dir = "./results/kaggle_house_price"

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# 获取数据集
	train_data = pd.read_csv(download('kaggle_house_train'))
	test_data = pd.read_csv(download('kaggle_house_test'))
	all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

	# 对所有的数值特征做归一化
	numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
	all_features[numeric_features] = all_features[numeric_features].apply(
		lambda x: (x - x.mean()) / (x.std()))

	# 填充缺失值
	all_features[numeric_features] = all_features[numeric_features].fillna(0)
	all_features = pd.get_dummies(all_features, dummy_na=True)

	# 获取训练集长度
	n_train = train_data.shape[0]
	train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
	test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
	train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)	# type:ignore

	loss = nn.MSELoss()
	in_features = train_features.shape[1]

	train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size, save_dir=save_dir)
	print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
		f'平均验证log rmse: {float(valid_l):f}')

	train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size, save_dir=save_dir)