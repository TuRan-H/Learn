/****************************************************************************************************
@author: TuRan
@data:	
@des:	不带头节点的单链表(LinkList whth No Head)
		不带头结点的单链表的头节点就是第一个元素
****************************************************************************************************/
#include "LinkListNH.h"


int main()
{
	NLinkList L;
	int i = 0;

	NLNode_Init(&L);

	NLNode_Creat_Tail(&L, scanf_string());

	NLNode_Delet(&L, 4, &i);

	NLNode_Show(L);
	
}


/**
 * @brief	初始化不带头结点的单链表
 * @param	L [NHLinkList *] 操作的单链表
 * @retval	bool
**/
bool NLNode_Init(NLinkList *L)
{
	(*L) = NULL;
	return TRUE;
}


/**
 * @brief	输出双链表
 * @param	L [NLinkLIst]
 * @retval	bool
**/
bool NLNode_Show(NLinkList L)
{
	printf("Your NLNode is : ");
	NLNode *p = L;
	while(p != NULL){
		printf("%d ", p->data);
		p = p->next;
	}
	putchar('\n');
}


/**
 * @brief	判断单链表是否为空
 * @param	L [NHLinkList]
 * @retval	treu == 为空 || false == 不为空
**/
bool NLNode_IfEmpty(NLinkList L)
{
	if(L == NULL) return TRUE;

	return FALSE;
}


/**
 * @brief	求表长
 * @param	L [NLinkList]
 * @retval	表的长度
**/
int NLNode_GetLength(NLinkList L)
{
	int i = 1;
	NLNode *p = L;

	while(p->next != NULL){
		i ++;
		p = p->next;
	}

	return i;
}


/**
 * @brief	修改节点上的内容
 * @param	L [LinkList *]
 * @retval	bool
**/
bool NLNode_Modify(NLinkList *L, int location, ElemType elem)
{
	if(location > NLNode_GetLength(*L) || (*L) == NULL) return FALSE;

	NLNode *p = NLNode_Find_Node(*L, location);
	p->data = elem;

	return TRUE;
}


/**
 * @brief	头插法创建双链表
 * @param	L [NLinkList]
 * @param	st [ElemType *] 需要插入双链表的数组内容
 * @retval	bool
**/
bool NLNode_Creat_Head(NLinkList *L, ElemType *st)
{
	NLNode *p = (*L);
	int i = 0;


	while(st[i] != '\0'){
		NLNode *s = (NLNode *)malloc(sizeof(NLNode));
		if(!s) return FALSE;
		s->data = st[i];

		s->next = (*L);
		(*L) = s;

		i ++;
	}

	return TRUE;
}


/**
 * @brief	尾插法创建双链表
 * @param	L [NLinkList]
 * @param	st [ElemType *]需要插入双链表的数组内容
 * @retval	bool
**/
bool NLNode_Creat_Tail(NLinkList *L, ElemType *st)
{
	NLNode *r;
	int i = 0;

	NLNode *q = (NLNode *)malloc(sizeof(NLNode));
	q->data = st[i];
	q->next = (*L);
	(*L) = q;
	r = q;
	i ++;

	if(st[i] != '\0'){
		while(st[i] != '\0'){
			NLNode *s = (NLNode *)malloc(sizeof(NLNode));
			s->data = st[i];

			s->next = r->next;
			r->next = s;
			r = s;

			i ++;
		}
	}else{
		return TRUE;
	}

	return TRUE;
}


/**
 * @brief	查找双链表的某一个节点
 * @param	L [NLinKList]
 * @param	location [int] 节点在双链表中的位置(从1开始)
 * @retval	NLNode *
**/
NLNode *NLNode_Find_Node(NLinkList L, int location)
{
	if(!L) return FALSE;

	NLNode *p = L;

	for(int i = 0; i < location	- 1; i ++){
		p = p->next;
		if(p == NULL) return NULL;
	}

	return p;
}


/**
 * @brief	按值查找
 * @param
 * @retval
**/
int NLNode_Find_ELem(NLinkList L, int elem)
{
	int i = 1;
	NLNode *p = L;
	while(p != NULL){
		if(p->data == elem) return i;
		i ++;
		p = p->next;
	}

	return -9999;
}


/**
 * @brief	按位查找
 * @param
 * @note
 * @example
 * @retval
**/
ElemType NLNode_Find_Location(NLinkList L, int location)
{
	NLNode *p =  NLNode_Find_Node(L, location);
	if(!p) return ERROR;

	return p->data;
}


/**
 * @brief	在双链表的某一个位置上插入一个节点
 * @param	L [NLinknList]
 * @param	location [int] 所需要插入的位置
 * @param	num [ElemType] 所插入位置的数据域
 * @example	location = 3 --> 在第三个位置上插入一个新的节点
 * @retval	bool
**/
bool NLNode_Insert(NLinkList *L, int location, ElemType num)
{
	NLNode *p = NLNode_Find_Node(*L, location - 1);
	NLNode *s = (NLNode *)malloc(sizeof(NLNode));
	s->data = num;

	s->next = p->next;
	p->next = s;

	return TRUE;
}


/**
 * @brief	在双链表的某一个节点后插入一个新的节点
 * @param	L [NLinkList]
 * @param	location [int] 被插入节点的位置
 * @param	num [ELemType] 新的节点的数据域
 * @retval	bool
**/
bool NLNode_Insert_Behind(NLinkList *L, int location, ElemType num)
{
	NLNode *p = NLNode_Find_Node(*L, location);
	NLNode *s = (NLNode *)malloc(sizeof(NLNode));
	s->data = num;

	s->next = p->next;
	p->next = s;

	return TRUE;
}


/**
 * @brief	在双链表的某一个节点前插入一个新的节点
 * @param	L [NLinkList *]
 * @param	p [NLNode *] 被插入节点的地址
 * @param	num [ElemType] 新的节点的数据域
 * @retval	bool
**/
bool NLNode_Insert_Front(NLinkList *L, NLNode *p, ElemType num)
{
	ElemType temp = 0;
	NLNode *s = (NLNode *)malloc(sizeof(NLNode));

	s->data = num;
	s->next = p->next;
	p->next = s;

	temp = p->data;
	p->data = s->data;
	s->data = temp;

	return TRUE;
}


/**
 * @brief	删除不带头结点的单链表上面的一个节点
 * @param	L [LinkList *]
 * @param	location [int]
 * @param	elem [ElemType *]
 * @retval	bool
**/
bool NLNode_Delet(NLinkList *L, int location, ElemType *elem)
{
	if(location > NLNode_GetLength(*L) || (*L) == NULL) return FALSE;

	NLNode *p = NLNode_Find_Node(*L, location - 1);
	*elem = p->next->data;
	NLNode *temp = p->next;

	p->next = p->next->next;
	free(temp);

	return TRUE;
}


/**
 * @brief	不带头结点的单链表的销毁
 * @param	L [NLinkList * ]
 * @retval	bool
**/
bool NLNode_Destory(NLinkList *L)
{
	if((*L) == NULL) return FALSE;

	NLNode *p = (*L);
	NLNode *temp = p;

	while(p != NULL){
		temp = p;
		p = p->next;
		free(temp);
	}

	return TRUE;
}