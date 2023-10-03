#ifndef __LINKQUEUE_H__
#define __LINKQUEUE_H__
#include "../../Lib/DAS.h"


typedef struct Queue_Node{
	ElemType data;
	struct Queue_Node *next;
}QNode;

typedef struct Linked_Queue{
	QNode *rear, *front;
}LinkQueue;


bool LinkQueue_EnQueue(LinkQueue *Q, ElemType elem);
bool LinkQueue_Init(LinkQueue *Q);
void LinkQueue_Show(LinkQueue Q);
bool LinkQueue_DeQueue(LinkQueue *Q, ElemType *elem);
bool LinkQueue_IfEmpty(LinkQueue Q);
bool LinkQueue_Destory(LinkQueue *Q);


#endif