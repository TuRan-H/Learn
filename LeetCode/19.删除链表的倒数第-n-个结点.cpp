// @before-stub-for-debug-begin
#include "commoncppproblem19.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=19 lang=cpp
 *
 * [19] 删除链表的倒数第 N 个结点
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

// 暴力解法
class Solution {
public:
	ListNode* removeNthFromEnd(ListNode* head, int n)
	{
		ListNode* p = head;

		// 列表的长度
		int size = 0;
		while (p != nullptr) {
			++size;
			p = p->next;
		}

		// 删除第一个节点
		if (n == size) {
			if (size != 1) {
				ListNode* q = head;
				head = head->next;
				delete q;
				return head;
				// ! 假设链表的长度为1, 删除一个元素直接返回nullptr
			} else {
				return nullptr;
			}
		}

		// 计算需要删除的节点
		int removed_node_index = size - n;

		// 将p指针移动到待删除节点的前一个节点
		p = head;
		for (int i = 0; i < removed_node_index - 1; ++i) {
			p = p->next;
		}

		// 删除p指针的下一个节点
		ListNode* q = p->next;
		p->next = q->next;
		delete q;

		return head;
	}
};
// @lc code=end
