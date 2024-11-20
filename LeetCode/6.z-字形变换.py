"""
主要是首先确定每一个元素应该所在的行号, 有了行号就简单了
"""


# @before-stub-for-debug-begin
from python3problem6 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode.cn id=6 lang=python3
#
# [6] Z 字形变换
#

# @lc code=start
class Solution:
	def convert(self, s: str, numRows: int) -> str:
		line_number_list = self.build_line_number(s, numRows)
		result_matrix = [list() for _ in range(numRows)]

		for line_number, element in zip(line_number_list, s):
			result_matrix[line_number].append(element)
		
		result = [result_element for result_line in result_matrix for result_element in result_line]

		return ''.join(result)

		
	def build_line_number(self, s: str, numRows: int):
		if numRows == 1:
			return [0] * len(s)

		line_number_list = list()
		line_number, line_number_increase_direction = 0, 1
		for _ in range(len(s)):
			if line_number == numRows-1:
				line_number_increase_direction = -1
			elif line_number == 0:
				line_number_increase_direction = 1

			line_number_list.append(line_number)
			line_number += line_number_increase_direction
		
		return line_number_list

		
		
# @lc code=end

