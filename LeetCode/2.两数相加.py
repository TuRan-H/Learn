# @before-stub-for-debug-begin
from python3problem2 import *
from typing import *
# @before-stub-for-debug-end

#
# @lc app=leetcode.cn id=2 lang=python3
#
# [2] 两数相加
#

# @lc code=start
# Definition for singly-linked list.
class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next

class Solution:
	def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode], carry: int = 0) -> Optional[ListNode]:  # # type: ignore
		if not l1 and not l2:
			return ListNode(carry) if carry != 0 else None

		if not l1: l1, l2 = l2, l1
		sum_result = l1.val + (l2.val if l2 else 0) + carry
		quotient = sum_result % 10
		remainder = sum_result // 10

		l1.val = quotient
		l1.next = self.addTwoNumbers(l1.next, l2.next if l2 else None, carry=remainder)

		return l1
# @lc code=end