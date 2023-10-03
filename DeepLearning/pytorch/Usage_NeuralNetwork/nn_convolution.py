"""
卷积(convolution)的使用方法, 本文主要着重于convolution 2d 的使用
详见: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
"""
import torch
import torch.nn.functional as f


input = torch.tensor(([1, 2, 0, 3, 1],  # 输入, 输入的是一个二维矩阵, 是一个tensor类型
					  [0, 1, 2, 3, 1],
					  [1, 2, 1, 0, 0],
					  [5, 2, 3, 1, 1],
					  [2, 1, 0, 1, 1]))

kernel = torch.tensor([[1, 2, 1],	# 卷积核, 卷积核是一个二维矩阵, 是一个tensor类型
					   [0, 1, 0],
					   [2, 1, 0]])

print(input.shape)	# 输出input的shape可以知道 他的shape格式为(5, 5), 表示五行五列
print(kernel.shape)	# 输出kernel的shape可以知道, 他的shape格式为(3, 3) 表示三行三列

"""
conv2d的input参数要求是(minibatch,in_channels,iH,iW)
weight参数的要求是(out_channels, in_channels/groups, kH, kW)
明显input和kernel的shape与conv2d的shape要求不同, 因此需要使用reshape函数
"""
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

"""
conv2d参数解析
input:输入
weight: 卷积核
stride: 步进
padding: 填充
输出: input和weight卷积结果
"""
output1 = f.conv2d(input=input, weight=kernel, stride=1, padding=0)
print(output1)
output2 = f.conv2d(input=input, weight=kernel, stride=2, padding=0)	# 测试stride = 2
print(output2)
output3 = f.conv2d(input=input, weight=kernel, stride=1, padding=1)	# 测试padding = 1
print(output3)