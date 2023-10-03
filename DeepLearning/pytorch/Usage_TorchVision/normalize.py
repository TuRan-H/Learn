"""
nomorlize 归一化 的实例文档
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

# 导入img图像
img_PIL = Image.open("../data/train/ants_image/0013035.jpg")

# 将PIL类型图像转化为tensor类型
tran_totensor = transforms.ToTensor()
img_tensor = tran_totensor(img_PIL)


# 归一化  normalize
trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
img_norm = trans_norm.forward(img_tensor)

# 将tensor类型的图像上传到Summary中
# 将normalize类型的图像上传到Summary中
writer = SummaryWriter("logs/event1")
writer.add_image("To Tensor", img_tensor)
writer.add_image("normalize", img_norm)
writer.close()