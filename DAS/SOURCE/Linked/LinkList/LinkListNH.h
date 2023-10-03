#ifndef __LINKLISTNH_H__
#define __LINKLISTNH_H__
#include "../../Lib/DAS.h"


// 单链表节点结构体
typedef struct Linked_Node_NoHead{
	ElemType data;
	struct Linked_Node_NoHead *next;
}NLNode, *NLinkList;


bool NLNode_Init(NLinkList *L);
bool NLNode_IfEmpty(NLinkList L);
bool NLNode_Creat_Head(NLinkList *L, ElemType *st);
bool NLNode_Show(NLinkList L);
bool NLNode_Creat_Tail(NLinkList *L, ElemType *st);
NLNode *NLNode_Find_Node(NLinkList L, int location);
bool NLNode_Insert(NLinkList *L, int location, ElemType num);
bool NLNode_Insert_Behind(NLinkList *L, int location, ElemType num);
bool NLNode_Insert_Front(NLinkList *L, NLNode *p, ElemType num);
int NLNode_GetLength(NLinkList L);
int NLNode_Find_ELem(NLinkList L, int elem);
ElemType NLNode_Find_Location(NLinkList L, int location);
bool NLNode_Destory(NLinkList *L);
bool NLNode_Modify(NLinkList *L, int location, ElemType elem);
bool NLNode_Delet(NLinkList *L, int location, ElemType *elem);


#endif