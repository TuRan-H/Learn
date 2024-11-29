# @before-stub-for-debug-begin
from python3problem7 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode.cn id=7 lang=python3
#
# [7] 整数反转
#

# @lc code=start
class Solution:
	def reverse(self, x: int) -> int:
		sign = 1 if x >= 0 else -1
		x *= sign

		str_x = str(x)
		reversed_x = int("".join(reversed(str_x))) * sign

		if reversed_x < -2**31 or reversed_x > 2**31-1:
			return 0

		return reversed_x
		
# @lc code=end

