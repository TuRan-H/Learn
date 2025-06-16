/*
和leetcode 24题差不多, leetcode24题相当于本体的k = 2

题解1:
设置滑动窗口, 窗口大小为k, 每次反转窗口内的所有元素, 进k步
 */

// @before-stub-for-debug-begin
#include "commoncppproblem25.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=25 lang=cpp
 *
 * [25] K 个一组翻转链表
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
	ListNode *reverseKGroup(ListNode *head, int k)
	{
		// 边界条件, 若 k = 1 则不需要进行反转
		if (k == 1)
			return head;

		// left首先指向头结点, right是left的后k-1个节点
		ListNode *left = head, *right = head;
		for (int i = 0; i < k - 1 && right->next != nullptr; i++)
			right = right->next;
		ListNode *new_head = right;
		// temp: 记录right后面的node, 用于left和right的更新
		ListNode *temp;
		// old_right, old_left: 用来记录本轮right和left的位置
		ListNode *old_right, *old_left;
		int count = 0;

		// 边界条件, 若链表的长度刚好等于k, 则直接翻转一次即可
		if (right->next == nullptr) {
			reverseList(left, right);
			return right;
		}

		while (left != right) {
			// 记录right后面一个节点的位置
			temp = right->next;
			// 记录本轮right和left的位置
			old_right = right;
			old_left = left;
			reverseList(left, right);
			// 更新left和right指针
			left = temp;
			right = left;
			// 如果left为空, 表示当前已经遍历结束了, 则无需再遍历了
			if(left == nullptr) {
				old_left->next = right;
				break;
			}
			for (count = 0; count < k - 1 && (right->next != NULL); ++count)
				right = right->next;
			// ! 如果left和right之间的差值不满足k, 则后面不再需要逆序
			if (count != k - 1) {
				old_left->next = left;
				break;
			}

			// 链接本轮的局部链表和下一轮的局部链表
			old_left->next = right;
		}

		return new_head;
	}

	// 反转从head -> tail的链表 (包括tail)
	void reverseList(ListNode *head, ListNode *tail)
	{
		ListNode *dummy_head = new ListNode(-1, head);

		if (!head || !head->next) {
			delete dummy_head;
			return;
		}

		ListNode *pre = dummy_head, *cur = head, *post = head->next;

		while (cur->next != tail) {
			cur->next = pre;
			pre = cur;
			cur = post;
			post = post->next;
		}

		cur->next = pre;
		tail->next = cur;
		head->next = nullptr;
		delete dummy_head;
	}
};
// @lc code=end
