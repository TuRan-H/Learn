"""
Resize 使用方法
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms


# 导入img文件, 作为PIL格式
img_path = "../data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)


# 将PIL类型转化为tensor类型
tran_ToTensor = transforms.ToTensor()	# 定义一个ToTensor的对象
img_ToTensor = tran_ToTensor(img_PIL)	# 调用ToTensor的__call__函数, 返回一个tensor类型

# resize
print(img_PIL.size)
trans_resize = transforms.Resize((512, 512))	# 定义一个Resize的对象
img_resize_PIL = trans_resize.forward(img_PIL)	# 调用Resize的forward函数, 将PIL类型的图片resize, 并且输出PIL类型的图片
img_resize_ToTensor = tran_ToTensor(img_resize_PIL)	# 将PIL类型转化为ToTensor类型
print(img_resize_PIL.resize)

# add_image
writer = SummaryWriter(log_dir="../runs/exp1")
writer.add_image("original", img_ToTensor)
writer.add_image("resize", img_resize_ToTensor)
writer.close()