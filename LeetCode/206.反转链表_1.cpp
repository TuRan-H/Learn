/*
通过递归实现反转列表
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
		// ! 最底层的递归出口 和 边界条件
		if (!head || !head->next) {
			return head;
		}

		// 让head节点的后面所有节点进行逆序
		ListNode *new_head = reverseList(head->next);

		// 使head节点的下一个节点的next指针指向head节点
		head->next->next = head;
		head->next = nullptr;

		return new_head;
	}
};
// @lc code=end
