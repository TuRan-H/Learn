import numpy as np

x = np.random.randn(10, 1, 28, 28)
# print(x[0][0][1])
for item in x[0][0][1]:
	print(item, end=' ')
