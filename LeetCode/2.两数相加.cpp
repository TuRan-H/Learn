// @before-stub-for-debug-begin
#include "commoncppproblem2.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=2 lang=cpp
 *
 * [2] 两数相加
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
	{
		ListNode* dummyHead = new ListNode(0);
		ListNode* current = dummyHead;
		int carry = 0;

		while (l1 != nullptr || l2 != nullptr || carry != 0) {
			int val1 = (l1 != nullptr) ? l1->val : 0;
			int val2 = (l2 != nullptr) ? l2->val : 0;

			int sum = val1 + val2 + carry;
			carry = sum / 10;
			int digit = sum % 10;

			current->next = new ListNode(digit);
			current = current->next;

			if (l1 != nullptr) {
				l1 = l1->next;
			}
			if (l2 != nullptr) {
				l2 = l2->next;
			}
		}

		ListNode* resultHead = dummyHead->next;
		delete dummyHead;

		return resultHead;
	}
};
// @lc code=end
