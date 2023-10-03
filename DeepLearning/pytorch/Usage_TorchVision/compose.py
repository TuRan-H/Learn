"""
compose 示例代码
compose 就是将多个变换过程组合起来, 一次性解决多个变换
假设有n个变换, n1, n2, n3.
其中后面一个变换的输入必须是前面一个变换的输出
i.e: n2的输入必须是n1的输出
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms


# 导入img文件, 作为PIL格式
img_path = "../data/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)


# 将PIL类型转化为tensor类型
trans_ToTensor = transforms.ToTensor()	# 定义一个ToTensor的对象
img_ToTensor = trans_ToTensor(img_PIL)	# 调用ToTensor的__call__函数, 返回一个tensor类型

# resize
trans_resize = transforms.Resize((512, 512))	# 定义一个Resize的对象
img_resize_PIL = trans_resize.forward(img_PIL)	# 调用Resize的forward函数, 将PIL类型的图片resize, 并且输出PIL类型的图片
img_resize_ToTensor = trans_ToTensor(img_resize_PIL)	# 将PIL类型转化为ToTensor类型

# compose
trans_resize_1 = transforms.Resize((512, 512))
trans_compose = transforms.Compose([trans_resize_1, trans_ToTensor])
"""
trans_resize_1 会将PIL类型的图像resize, 转换成PIL_resize
trans_ToTensor输入PIL_resize将其转话成Tensor_resize
"""
img_resize_ToTensor_1 = trans_compose(img_PIL)	# 输入结果是一个resize后的tensor类型

# write
write = SummaryWriter(log_dir="runs/exp2")
write.add_image("original", img_ToTensor)
write.add_image("resize", img_resize_ToTensor_1)
write.close()