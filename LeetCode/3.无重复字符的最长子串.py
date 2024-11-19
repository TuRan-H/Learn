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
		D = [0] * len(s)

		for i in range(len(s)):
			if i == 0:
				D[i] = 1
			elif s[i] == s[i-1]:
				D[i] = D[i-1]
			else:   # s[i] != s[i-1]
				# find j
				j = -1
				for index in range(i-1, -1, -1):
					if s[index] == s[i]:
						j = index
						break
				
				# find len(i and j loop back until s[i] != s[j])
				count = 0
				if j != -1:
					i_temp, j_temp = i, j
					while(s[i_temp] == s[j_temp]):
						count += 1
						i_temp -= 1
						j_temp -= 1
				
				D[i] = min(count, D[i-1]+1) if count != 0 else D[i-1]+1

		return max(D)
		
# @lc code=end