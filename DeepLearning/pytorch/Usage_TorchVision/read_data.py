from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
	def __init__(self, root_dir, label_path):
		self.root_dir = root_dir		# root_dir 表示根目录的名称
		self.label_dir = label_path		# label_dir 表示标签名称
		self.path = os.path.join(root_dir, label_path)
		self.img_path = os.listdir(self.path)		# img_path 表示root_dir 和 label_dir 组合而成的目录下面的所有文件的一个文件名列表

	def __getitem__(self, index):
		img_name = self.img_path[index]
		img_item_path = os.path.join(self.path, img_name)
		img = Image.open(img_item_path)
		label = self.label_dir
		return img, label

	def __len__(self):
		return len(self.img_path)


root_path = "dataset\\train"
ants_label_path = "ants"
ants_dataset = MyData(root_path, ants_label_path)
img_test, label_test = ants_dataset[0]
