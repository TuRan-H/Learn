/****************************************************************************************************
@author: TuRan
@des: 	顺序队列相关代码
		Sequence queue code
		front	队头指针 指向最后一个元素, 用于出队
		rear	队尾指针 指向队尾元素的下一个元素, 用于入队
****************************************************************************************************/
#include "SqQueue.h"


int main(int argc, char *argv[])
{
	SqQueue Q;
	ElemType temp;
	

	SqQueue_Init(&Q);


	SqQueue_EnQueue(&Q, 3);
	SqQueue_EnQueue(&Q, 3);
	SqQueue_EnQueue(&Q, 3);
	SqQueue_EnQueue(&Q, 4);

	printf("%d", SqQueue_GetTop(Q));
	putchar('\n');







	SqQueue_Show(Q);
}


/**
 * @brief	顺序队列初始化函数
 * @param	Q [SqQueue *] 
 * @retval	bool
*/
bool SqQueue_Init(SqQueue *Q)
{
	Q->data = (ElemType *)malloc(sizeof(ElemType) * INITSIZE);
	if(!Q) return FALSE;
	Q->front = Q->rear = 0;

	return TRUE;
}


/**
 * @brief	顺序队列输出函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqQueue_Show(SqQueue Q)
{
	int i = Q.front;

	while(i != Q.rear){
		printf("%d ", Q.data[i]);

		(++ i ) % MAXSIZE;
	}

	return TRUE;
}


/**
 * @brief	顺序队列入队函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqQueue_EnQueue(SqQueue *Q, ElemType elem)
{
	if(SqQueue_IfFull(*Q)) return FALSE;

	Q->data[Q->rear] = elem;
	Q->rear = (Q->rear + 1) % MAXSIZE;

	return TRUE;
}


/**
 * @brief	顺序队列出队函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqQueue_DeQueue(SqQueue *Q, ElemType *elem)
{
	if(SqQueue_IfEmpty(*Q)) return FALSE;
	*elem = Q->data[Q->front];

	Q->front = (Q->front + 1) % MAXSIZE;
}


/**
 * @brief	顺序队列求队长函数
 * @param
 * @note
 * @example
 * @retval
**/
int SqQueue_GetLength(SqQueue Q)
{
	int length = 0;
	length = (Q.rear + MAXSIZE - Q.front) % MAXSIZE;

	return length;
}


/**
 * @brief	顺序队列判断队满函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqQueue_IfFull(SqQueue Q)
{
	if((Q.rear + 1) % MAXSIZE == Q.front) return TRUE;

	return FALSE;
}


/**
 * @brief	顺序队列判断队列空函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqQueue_IfEmpty(SqQueue Q)
{
	if(Q.rear == Q.front) return TRUE;

	return FALSE; 
}


/**
 * @brief	顺序队列销毁代码
 * @param
 * @note
 * @example
 * @retval
**/
bool SqQueue_Destory(SqQueue *Q)
{
	Q->front = Q->rear = 0;
	free(Q->data);

	return TRUE;
}


/**
 * @brief
 * @param
 * @note
 * @example
 * @retval
**/
ElemType SqQueue_GetTop(SqQueue Q)
{
	if(SqQueue_IfEmpty(Q)) return ERROR;

	return Q.data[Q.rear - 1];
}