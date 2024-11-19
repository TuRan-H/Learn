# @before-stub-for-debug-begin
from python3problem1 import *
from typing import *
# @before-stub-for-debug-end


#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
	def twoSum(self, nums: List[int], target: int) -> List[int]:
		return_index = []

		for index, i in enumerate(nums):
			for j in nums[index+1:]:
				if i +j == target: 
					return_index.append(nums.index(i))
					nums.remove(i)
					return_index.append(nums.index(j) + 1)
		return return_index
		
# @lc code=end


# #
# @lc app=leetcode.cn id=1 lang=python
#
# [1] 两数之和
#

