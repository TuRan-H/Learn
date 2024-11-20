# @before-stub-for-debug-begin
from python3problem4 import *
from typing import * # type: ignore
# @before-stub-for-debug-end


#
# @lc app=leetcode.cn id=4 lang=python3
#
# [4] 寻找两个正序数组的中位数
#
# @lc code=start
class Solution:
	def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
		i, j = 0, 0
		merge_list = list()

		while i != len(nums1) and j != len(nums2):
			if nums1[i] <= nums2[j]:
				merge_list.append(nums1[i])
				i += 1
			else:
				merge_list.append(nums2[j])
				j += 1
		
		if i == len(nums1): merge_list.extend(nums2[j:])
		else: merge_list.extend(nums1[i:])

		length = len(merge_list)
		if length % 2 == 0:
			median_value = (merge_list[length // 2 - 1] + merge_list[length // 2]) / 2
		else:
			median_value = merge_list[length // 2]

		return median_value
		
# @lc code=end

