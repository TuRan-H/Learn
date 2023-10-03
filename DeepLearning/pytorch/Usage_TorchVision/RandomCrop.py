"""
RandomCrop 随机裁剪
设置一个裁剪参数, 将一个图片随机裁剪成制定参数的大小
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms


# 导入图片
img_path = "../data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
trans_ToTensor = transforms.ToTensor()
img_ToTensor = trans_ToTensor(img_PIL)


# Random Crop
trans_RandomCrop = transforms.RandomCrop(50)
trans_compose = transforms.Compose([trans_RandomCrop,trans_ToTensor])


# add image
writer = SummaryWriter(log_dir="../runs/exp1")
writer.add_image("original", img_ToTensor)
for i in range(10):
	img_RandomCrop = trans_compose(img_PIL)
	writer.add_image("RandomCrop", img_RandomCrop, i)

writer.close()
