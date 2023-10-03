#ifndef __LINKLIST_H__
#define __LINKLIST_H__
#include "../../Lib/DAS.h"


// 单链表节点结构体
typedef struct Linked_Node{
	ElemType data;
	struct Linked_Node *next;
}LNode, *LinkList;


bool LNode_Creat_Head(LinkList *L, ElemType *st_data);
bool LNode_Init(LinkList *L);
void LNode_Show(LinkList L);
ElemType LNode_Find_Location(LinkList L, int location);
bool LNode_Modify(LinkList L, int location, ElemType num);
bool LNode_Insert(LinkList *L, int location, ElemType data);
bool LNode_Insert_Behind(LinkList *L, int locatoin, ElemType num);
bool LNode_Insert_Front(LinkList *L, LNode *p, ElemType data);
bool LNode_Delet(LinkList *L, int location, ElemType *num);
int LNode_Find_Elem(LinkList L, ElemType elem);
LNode *LNode_Find_Node(LinkList L, int location);
bool LNode_Creat_Tail(LinkList *L, ElemType *st_data);
bool LNode_IfEmpty(LinkList L);
int LNode_GetLength(LinkList L);
bool LNode_Destory(LinkList *L);


#endif