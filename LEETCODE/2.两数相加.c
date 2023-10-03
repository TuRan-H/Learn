/*
 * @lc app=leetcode.cn id=2 lang=c
 *
 * [2] 两数相加
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 */


// struct ListNode {
//     int val;
//     struct ListNode *next;
// };




struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2){
	struct ListNode *p = l1,  *q = l2;	// 创建两个指针, p指向l1, q指向l2
	struct ListNode *l = (struct ListNode*) malloc(sizeof(struct ListNode));	// 创建了一个新的链表用作返回
	l->next = NULL;	// 将链表的next域置空
	struct ListNode *r = l;	// 创建指针r指向链表l
	int carry = 0;	// 进位标志
	int addresult = 0;	// 每次让指针p和指针q后移一位, addresult是指针p和q的和
	int n1 = 0, n2 = 0;	// n1 = 指正p所指向元素的值域, n2 = 指针q所指向元素的值域
	while(p || q){
		if(p) n1 = p->val; else n1 = 0;		// 假设指针p不存在, 则让n1 = 0, 否则让n1 = 指针p所指向的值域
		if(q) n2 = q->val; else n2 = 0;
		addresult = n1 + n2 + carry;
		if(addresult >= 10){
			addresult -= 10;
			carry = 1;
		}else carry = 0;
		r->val = addresult;
		if((p && p->next) || (q && q->next)){		// 假设指针p或者指针q有一个存在, 则建立l链表的下一个节点
			r->next = (struct ListNode *)malloc(sizeof(struct ListNode));
			r->next->next = NULL;
			r = r->next;
		} 
		if(p && p->next) p = p->next; else p = NULL;	// 结尾判断条件, 讲治镇p和q置空
		if(q && q->next) q = q->next; else q = NULL;
	}
	if(carry == 1){
		r->next = (struct ListNode *)malloc(sizeof(struct ListNode));	// 假设在处理完毕后carry = 1, 创建一个新的节点使节点值 = 1
		r->next->next = NULL;
		r->next->val = 1;
	}

	return l;
}




// @lc code=end

