#ifndef __DLINKLIST_H__
#define __DLINKLIST_H__
#include "../../Lib/DAS.h"


// 双链表节点结构体
typedef struct Dual_Linked_Node{
	ElemType data;
	struct Dual_Linked_Node *next;
	struct Dual_Linked_Node *prior;
}DLNode, *DLinkList;


bool DLNode_Init(DLinkList *L);
void DLNode_Show(DLinkList L);
void DLNode_Show_Reverse(DLinkList L);
bool DLNode_Creat_Tail(DLinkList *L, ElemType *st_data);
bool DLNode_Creat_Head(DLinkList *L, ElemType *st_data);
bool DLNode_Insert_Behind(DLinkList *L, int location, ElemType num);
bool DLNode_Insert_Front(DLinkList *L, int location, ElemType elem);
bool DLNode_Delet(DLinkList *L, int location, ElemType *num);
bool DLNode_Modify(DLinkList *L, int location, ElemType num);
ElemType DLNode_Find_Location(DLinkList L, int location);
int DLNode_Find_Elem(DLinkList L, ElemType elem);
DLNode * DLNode_Find_Node(DLinkList L, int location);
bool DLNode_Destory(DLinkList *L);
int DLNode_GetLength(DLinkList L);
bool DLNode_IfEmpty(DLinkList L);
bool DLNode_Insert(DLinkList *L, int location, ElemType elem);


#endif
