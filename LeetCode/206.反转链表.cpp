/*
通过迭代实现反转列表
 */

// @before-stub-for-debug-begin
#include "commoncppproblem206.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=206 lang=cpp
 *
 * [206] 反转链表
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
	ListNode *reverseList(ListNode *head)
	{
		// 假头结点
		ListNode *dummy_head = new ListNode(-1, head);

		// 边界条件, 链表中仅有一个节点 or 链表中一个节点都没有
		// ! 必须要先处理head, 再处理head->next, 因为不确定head->next是否存在
		if (!head || !head->next) {
			delete dummy_head;
			return head;
		}

		ListNode *pre = dummy_head, *cur = head, *post = head->next;

		while (cur->next) {
			// 当前节点的next指向其前一个节点
			cur->next = pre;
			pre = cur;
			cur = post;
			post = post->next;
		}

		cur->next = pre;
		head->next = nullptr;
		delete dummy_head;

		return cur;
	}
};
// @lc code=end
