"""
思路: 以此遍历所有元素 index, 利用回文数的特性, 使用双指针判断index周围的最大回文子串
视频链接: https://www.bilibili.com/video/BV1UZ4y1U7tt/?spm_id_from=333.337.search-card.all.click&vd_source=41721633578b9591ada330add5535721
"""


# @before-stub-for-debug-begin
from python3problem5 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode.cn id=5 lang=python3
#
# [5] 最长回文子串
#

# @lc code=start
class Solution:
	def longestPalindrome(self, s: str) -> str:
		longest_length, longest_palindrome = 0, str()

		# ODD
		for index in range(len(s)):
			left, right = index, index
			while (left >= 0 and right < len(s)) and (s[left] == s[right]):
				if right-left+1 > longest_length:
					longest_length = right-left+1
					longest_palindrome = s[left:right+1]
				left -= 1
				right += 1
		
		# EVEN
		for index in range(len(s)):
			left, right = index, index+1
			while (left >= 0 and right < len(s)) and (s[left] == s[right]):
				if right-left+1 > longest_length:
					longest_length = right-left+1
					longest_palindrome = s[left:right+1]
				left -= 1
				right += 1
		
		return longest_palindrome

		
# @lc code=end

