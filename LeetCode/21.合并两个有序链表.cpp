/*
回溯算法
 */
// @before-stub-for-debug-begin
#include "commoncppproblem21.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=21 lang=cpp
 *
 * [21] 合并两个有序链表
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
	ListNode *mergeTwoLists(ListNode *list1, ListNode *list2)
	{
		ListNode *head = new ListNode(-1, nullptr);
		ListNode *tail = head;

		while (list1 && list2) {
			if (list1->val < list2->val) {
				tail->next = list1;
				list1 = list1->next;
			} else {
				tail->next = list2;
				list2 = list2->next;
			}
			tail = tail->next;
		}

		tail->next = (list1) ? list1 : list2;

		return head->next;
	}
};
// @lc code=end
