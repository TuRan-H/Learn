// @before-stub-for-debug-begin
#include "commoncppproblem24.h"
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=24 lang=cpp
 *
 * [24] 两两交换链表中的节点
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
	ListNode *swapPairs(ListNode *head)
	{
		ListNode *dummy_head = new ListNode(-1, head);
		ListNode *cur = dummy_head;
		ListNode *p, *q;

		// 如果链表中没有节点 或者链表中只有一个节点, 则直接返回head
		if (!head)
			return head;
		else if (!head->next)
			return head;

		// 如果链表中仅有一个节点
		p = head;
		q = head->next;

		while (p && q) {
			cur->next = q;
			p->next = q->next;
			q->next = p;

			cur = p;
			p = cur->next;
			if (p)
				q = p->next;
			else
				q = nullptr;
		}

		return dummy_head->next;
	}
};
// @lc code=end
