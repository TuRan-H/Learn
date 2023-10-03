"""
使用nn_module创建一个自己的类(示例)
"""
import torch


class TuRan(torch.nn.Module):
	"""
	集成与 nn.Module的一个子类, 该子类必overwrite forward函数
	forward函数: 类似于__call__函数, 让我们可以你直接调用实例
	"""
	def __init__(self):
		super().__init__()

	def forward(self, input):
		output = input + 1
		return output

turan = TuRan()
x = torch.tensor(1)
y = 1
output = turan(y)
print(output)