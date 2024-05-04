"""
最长递增子序列算法 Logest increase subsequence algorithm
"""
import numpy as np


def longest_increasing_subsequence(a:list) -> tuple[int, list]:
	"""
	找到给定列表的最长递增子序列（LIS）。

	Args:
		a (list): 输入列表。

	Returns:
		tuple[int, list]: 包含LIS的长度和LIS本身的元组。

	Examples:
		>>> a = [3, 2, 5, 1, 6]
		>>> longest_increasing_subsequence(a)
		(3, [2, 5, 6])
	"""
	# 获取列表长度
	n = len(a)
	# 创建一个长度为n的全1数组，用于存储LIS的长度
	dp = np.ones(n, dtype=int)
	# 创建一个长度为n的数组，用于存储每个元素的前一个元素的索引
	prev = -1 * dp
	# 创建一个空列表，用于存储LIS
	sequence = list()

	# 遍历列表中的每个元素
	for i in range(n):
		temp = 0
		for j in range(i):
			if a[j] < a[i] and temp < dp[j]:
				temp = dp[j]
				prev[i] = j
		dp[i] = temp + 1

	# 找到LIS的最后一个元素的索引
	i = np.argmax(dp)
	# 从最后一个元素开始，依次将元素添加到LIS中
	while i != -1:
		sequence.append(a[i])
		i = prev[i]

	# 反转LIS列表
	sequence.reverse()

	# 返回LIS的长度和LIS本身
	return np.argmax(dp), sequence


if __name__ == "__main__":
	a = [1,5,3,4,8]

	print(longest_increasing_subsequence(a))