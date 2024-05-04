/***************************************************************************************************
@author: TuRan
@data:	2021/5/21
@des: 	带头结点的双链表
		链表的变体 --> 双链表: 即除了next域之外多了一个prior域
		双链表相关代码
		Dual linked list code
****************************************************************************************************/
// #define _CRT_SECURE_NO_WARNINGS
#include "DLinkList.h"


int main(int argc, char *argv[])
{
	ElemType temp;
	DLinkList L;
	DLNode_Init(&L);

	DLNode_Creat_Tail(&L, scanf_string());


	DLNode_Insert(&L, 3, 9);


	DLNode_Show(L);


}


/**
 * @brief	初始化双链表
 * @param	L [DLinkList *]
 * @retval	bool
**/
bool DLNode_Init(DLinkList *L)
{
	(*L) = (DLNode *)malloc(sizeof(DLNode));
	if(!(*L)) return FALSE;

	(*L)->next = NULL;
	(*L)->data = 0;

	printf("Init Success\n");

	return TRUE;
}


/**
 * @brief	双链表输出链表
 * @param	L [DLinkList]
 * @retval	void
**/
void DLNode_Show(DLinkList L)
{
	printf("Your DLNode is :\n");
	DLNode *p = L->next;
	while(p != NULL){
		printf("%d ", p->data);
		p = p->next;
	}
	putchar('\n');
}


/**
 * @brief	双链表判空
 * @param	L [DLinkList]
 * @retval	TURE | FALSE
**/
bool DLNode_IfEmpty(DLinkList L)
{
	if(L->next == NULL) return TRUE;

	return FALSE;	
}


/**
 * @brief	求双链表表长
 * @param	L [DLinkList]
 * @retval	bool
**/
int DLNode_GetLength(DLinkList L)
{
	int length = 0;
	for(DLNode *p = L->next; p != NULL; p = p->next){
		length ++;
	}

	return length;
}


/**
 * @brief	双链表输出链表(反转输出)
 * @param	L [DLinkList L]
 * @retval	void
**/
void DLNode_Show_Reverse(DLinkList L)
{
	printf("Your DLNode reverse is :\n");
	DLNode *p = L->next;
	while(p->next != NULL){
		p = p->next;
	}
	while(p != L){
		printf("%d ", p->data);
		p = p->prior;
	}
	putchar('\n');
}


/**
 * @brief	删除 删除双链表的某一个节点, 并返回该节点的数据域内容
 * @param	L [DLinkList *] 进行操作双链表
 * @param	location [int] 需要删除节点的位置(从1开始)
 * @param	num [ElemType *] 被删除的数据域内容
 * @retval	bool
**/
bool DLNode_Delet(DLinkList *L, int location, ElemType *num)
{
	if(!(*L)) return FALSE;
	DLNode *p = DLNode_Find_Node(*L, location);
	DLNode *temp;

	

	*num = p->data;

	p->prior->next = p->next;
	if(p->next != NULL){
		p->next->prior = p->prior;
	}

	free(p);

	return TRUE;
}


/**
 * @brief	(改) 修改双链表某一个节点上面的数值
 * @param	L [DLinkList *] 需要修改的双链表
 * @param	location [int] 需要修改的节点的位置(从1开始)
 * @param	num [ElemType] 需要修改的数据
 * @retval	bool
**/
bool DLNode_Modify(DLinkList *L, int location, ElemType num)
{
	DLNode *p = DLNode_Find_Node(*L, location);
	if(!(*L)) return FALSE;


	p->data = num;

	return TRUE;
}


/**
 * @brief	双链表使用头插法创建链表
 * @param	L [DLinkList *]
 * @param	st_data [ElemType *] 单链表的内容来源(数组)
 * @retval	bool
**/
bool DLNode_Creat_Head(DLinkList *L, ElemType *st_data)
{
	if(*L == NULL) return FALSE;
	int i = 0;

	while(st_data[i] != '\0'){
		DLNode *s = (DLNode *)malloc(sizeof(DLNode));
		
		if(!s) return FALSE;

		s->next = (*L)->next;			// 新的节点next指向第一个节点的后面一个节点
		if((*L)->next != NULL){			// 假设第一个节点的后面一个节点不为空, 让第二个节点和第一个节点连接起来
			(*L)->next->prior = s;		
		}
		(*L)->next = s;
		s->prior = *L;					// 链接新的节点和第一个节点
		s->data = st_data[i];

		i ++;
	}

	return TRUE;
}


/**
 * @brief	双链表使用尾插法创建链表
 * @param	L [DLinkList *]
 * @param	st_data [ElemType *] 单链表的内容来源(数组)
 * @retval	bool
**/
bool DLNode_Creat_Tail(DLinkList *L, ElemType *st_data)
{
	DLNode *r = (*L);
	int i = 0;

	if((*L) == NULL) return FALSE;

	while(st_data[i] != '\0'){
		DLNode *s = (DLNode *)malloc(sizeof(DLNode));
		if(!s) return FALSE;
		s->data = st_data[i];
		s->next = r->next;
		r->next = s;
		s->prior = r;
		r = s;

		i ++;
	}

	return TRUE;
}


/**
 * @brief	在单链表的某一个位置上插入一个新的节点
 * @param	L [DLinkList]
 * @param	location [int]
 * @param	elem [ElemType]
 * @retval	TURE | FLASE
**/
bool DLNode_Insert(DLinkList *L, int location, ElemType elem)
{
	if(location > DLNode_GetLength(*L)) return FALSE;

	DLNode *p = DLNode_Find_Node(*L, location - 1);
	DLNode *s = (DLNode *)malloc(sizeof(DLNode));
	s->data = elem;

	s->next = p->next;
	s->prior = p;
	p->next->prior = s;
	p->next = s;

	return TRUE;
}


/**
 * @brief	(增) 双链表在某一个节点后面增加一个节点
 * @param	L [DLinkList *] 需要修改的双链表
 * @param	location [int] 插入节点的位置, 从1开始
 * @param	num [ElemType] 新建节点的数据域内容
 * @retval	bool
**/
bool DLNode_Insert_Behind(DLinkList *L, int location, ElemType num)
{
	DLNode *s = (DLNode *)malloc(sizeof(DLNode));
	if(!(*L)) return FALSE;
	DLNode *p = DLNode_Find_Node(*L, location);

	
	s->data = num;
	if(p->next != NULL){
		p->next->prior = s;
	}
	s->next = p->next;
	p->next = s;
	s->prior = p;

	return TRUE;
}


/**
 * @brief	在某一个节点前面添加一个节点
 * @param	L [DLinkList *]
 * @param	location [int] 节点在双年表中的位置(从1开始)
 * @param	elem [ElemType] 新节点的data
 * @retval	bool
**/
bool DLNode_Insert_Front(DLinkList *L, int location, ElemType elem)
{
	if(L == NULL) return FALSE;
	DLNode *p = DLNode_Find_Node(*L, location - 1);

	if(!p) return FALSE;

	DLNode *s = (DLNode *)malloc(sizeof(DLNode));

	s->data = elem;
	s->next = p->next;
	p->next->prior = s;
	p->next = s;
	s->prior = p;

	return TRUE;
}


/**
 * @brief	按位查找 查询当前双链表某一个节点的数据域, 根据节点位置查找
 * @param	L [DLinkList] 需要查询的双链表
 * @param	location [int] 需要查询的双链表的节点的位置(从1开始)
 * @retval	ElemType 查询到的双链表节点的数据域内容
**/
ElemType DLNode_Find_Location(DLinkList L, int location)
{
	DLNode *p = L->next;
	for(int i = 0; i < location; i ++){
		p = p->next;
		if(p == NULL) return FALSE;
	}

	return p->data;
}


/**
 * @brief	(查)根据给出的元素查询该元素在双年表中的位置
 * @param	L [DLinkList] 被查询的双链表
 * @param	elem [ElemType] 给出的元素
 * @retval	该元素在双链表中的位置, 从1开始
**/
int DLNode_Find_Elem(DLinkList L, ElemType elem)
{
	DLNode *p = L->next;
	int i = 1;

	while(p != NULL){
		if(p->data == elem) return i;
		
		i ++;
		p = p->next;
	}
	

	return -9999;
}


/**
 * @brief	查找双链表某一个位置上的节点
 * @param	L [DLinkList] 查找的对象
 * @param	location [int] 该节点在双链表上的位置, 从1开始
 * @retval	查找到的节点的指针
**/
DLNode * DLNode_Find_Node(DLinkList L, int location)
{
	DLNode *p = L;			// 令p节点指向第一个节点
	for(int i = 0; i < location; i ++){
		p = p->next;
		if(p == NULL){
			return NULL;
		}
	}


	return p;
}


/**
 * @brief	双链表销毁代码
 * @param	L [LinkList *]
 * @retval	bool
**/
bool DLNode_Destory(DLinkList *L)
{
	if(!(*L)) return FALSE;

	DLNode *p = (*L)->next;
	DLNode *temp = p;

	while(p != NULL){
		temp = p;
		p = p->next;
		free(temp);
	}

	return TRUE;
}