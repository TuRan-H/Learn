#ifndef __SQQUEUE_H__
#define __SQQUEUE_H__
#include "../../Lib/DAS.h"


// 顺序队列结构体
typedef struct Sequence_Queue{
	ElemType *data;
	int front;		// 对头
	int rear;		// 队尾
}SqQueue;


bool SqQueue_Init(SqQueue *Q);
bool SqQueue_EnQueue(SqQueue *Q, ElemType elem);
bool SqQueue_Show(SqQueue Q);
int SqQueue_GetLength(SqQueue Q);
bool SqQueue_DeQueue(SqQueue *Q, ElemType *elem);
bool SqQueue_IfFull(SqQueue Q);
bool SqQueue_Destory(SqQueue *Q);
bool SqQueue_IfEmpty(SqQueue Q);
ElemType SqQueue_GetTop(SqQueue Q);


#endif