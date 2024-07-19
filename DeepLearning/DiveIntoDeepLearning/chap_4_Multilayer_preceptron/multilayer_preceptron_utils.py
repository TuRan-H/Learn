import random
import torch
import matplotlib.pyplot as plt




def FashionMnist_id2label(labels):
	"""
	根据给出的labels, 获取真实的labels

	note
	---
	输入labels是一个列表, 列表中的每一个元素都是int类型, 表示一个类别号
	比如说labels[0] = 1, 那么就说明第0号样本的标签为trouser

	params
	---
	labels: 使用数字表示的标签列表
	"""
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
				   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]


def evaluate_MLP(net, test_iter, batch_size, device=None, show_title=None):
	"""
	评估MLP的准确性

	Args:
	---
		net: 神经网络模型, 需要有__call__() / forward()方法
		test_iter: 测试数据集的迭代器
		batch_size: 批量大小
		device: 使用的device
		show_title: 是否使用文字形式来显示预测值和真值的title
	"""
	index = [random.randint(0, batch_size) for _ in range(10)]
	with torch.no_grad():
		x, y = next(iter(test_iter))
		x, y = x[index], y[index]
		x_origin = x
		x = x.reshape(10, -1)
		if device:
			x = x.to(device)
			y = y.to(device)
			net = net.to(device)
		y_hat = net(x)
		y_hat = torch.argmax(y_hat, dim=1)
		if show_title:
			y_hat_title = FashionMnist_id2label(y_hat)
			y_title = FashionMnist_id2label(y)
		show_images(x_origin, 2, 5, titles=y, titles_pred=y_hat)
	plt.show()


def show_images(imgs, num_rows, num_cols, titles=None, titles_pred=None, scale=2):
	"""
	显示一系列的图像

	Args:
	---
		imgs: matplotlib支持的图像数据, shape=(H, W, C)
		num_rows: subplot的行数
		num_clos: subplot的列数
		titles: 真实的标签
		titles_pred: 预测出来的标签
		scale: figure的大小
	Returns:
	---

	"""
	figsize = (num_cols*scale, num_rows*scale)
	_, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
	# 调整子图之间的距离
	plt.subplots_adjust(wspace=0.5, hspace=0.5)
	axes = axes.flatten()
	# ! 一个多值的张量不能使用 if title来判定是否为空
	if titles is not None:
		titles = FashionMnist_id2label(titles)
	if titles_pred is not None:
		titles_pred = FashionMnist_id2label(titles_pred)
	for i, (ax, img) in enumerate(zip(axes, imgs)):
		try:
			ax.imshow(img)
		except:
			img = img.numpy()
			# ! matplotlib接受的图像shape应该是(H, W, C), 而fashion取出来是(C, H, W)
			img = img.transpose(1, 2, 0)
			ax.imshow(img)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.set_title(titles[i])
		if titles_pred:
			ax.set_title(f"{titles[i]}\n{titles_pred[i]}")
	return axes