"""
求解思路: 滑动窗口

参考资料: https://www.bilibili.com/video/BV113411v7Ak/?spm_id_from=333.337.search-card.all.click&vd_source=41721633578b9591ada330add5535721
"""


# @before-stub-for-debug-begin
from python3problem3 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode.cn id=3 lang=python3
#
# [3] 无重复字符的最长子串
#

# @lc code=start
class Solution:
	def lengthOfLongestSubstring(self, s: str) -> int:
		length, max_length = 0, 0
		right, left = 0, 0
		unique_element = set()

		while right != len(s):
			if s[right] not in unique_element:
				unique_element.add(s[right])
				right += 1
			else:
				unique_element.discard(s[left])
				left += 1
			
			length = len(unique_element)
			if length > max_length: max_length = length
		
		return max_length


		
# @lc code=end