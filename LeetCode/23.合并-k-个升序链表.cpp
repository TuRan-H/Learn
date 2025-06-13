/*
分治算法

先将问题拆分为多个子问题, 再将子问题进行拆分, 直到不可拆分
在不可拆分时, 解决当前最小的子问题
两两合并子问题
 */

// @before-stub-for-debug-begin
#include "commoncppproblem23.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=23 lang=cpp
 *
 * [23] 合并 K 个升序链表
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
	ListNode *mergeKLists(vector<ListNode *> &lists)
	{
		if (lists.size() == 0)
			return nullptr;

		return merge(lists, 0, lists.size() - 1);
	}

private:
	ListNode *merge(vector<ListNode *> lists, int l, int r)
	{
		if (l == r)
			return lists[l];

		int mid = (l + r) / 2;

		return merge_two_list(merge(lists, l, mid), merge(lists, mid + 1, r));
	}

	ListNode *merge_two_list(ListNode *list_1, ListNode *list_2)
	{
		ListNode *head = new ListNode(-1, nullptr);
		ListNode *tail = head;

		while (list_1 && list_2) {
			if (list_1->val < list_2->val) {
				tail->next = list_1;
				list_1 = list_1->next;
			} else {
				tail->next = list_2;
				list_2 = list_2->next;
			}
			tail = tail->next;
		}

		tail->next = (list_1) ? list_1 : list_2;

		return head->next;
	}
};
// @lc code=end
