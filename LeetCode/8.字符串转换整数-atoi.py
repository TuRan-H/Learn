# @before-stub-for-debug-begin
from python3problem8 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode.cn id=8 lang=python3
#
# [8] 字符串转换整数 (atoi)
#

# @lc code=start
class Solution:
	def myAtoi(self, s: str) -> int:
		# 去除前导空格
		s = s.strip(" ")
		if s == "": return 0
		# 判断正负号
		sign = 1
		if s[0] == '-':
			s = s[1:]
			sign = -1
		elif s[0] == "+":
			s = s[1:]
		# 解析字符串
		resolved_int, max_int, min_int, boundary = 0, 2**31-1, -2**31, (2**31-1)//10
		for elem in s:
			if elem not in "0123456789": break
			if resolved_int > boundary or resolved_int == boundary and elem > '7':
				return max_int if sign ==1 else min_int
			resolved_int = 10*resolved_int+int(elem)

		return sign * resolved_int
# @lc code=end