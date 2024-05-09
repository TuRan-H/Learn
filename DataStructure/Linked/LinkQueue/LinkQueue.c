#include "LinkQueue.h"


int main(int argc, char *argv[])
{
	LinkQueue Q;
	ElemType temp;


	LinkQueue_Init(&Q);

	LinkQueue_EnQueue(&Q, 3);
	LinkQueue_EnQueue(&Q, 3);
	LinkQueue_EnQueue(&Q, 3);
	LinkQueue_EnQueue(&Q, 3);
	LinkQueue_EnQueue(&Q, 3);

	LinkQueue_Destory(&Q);


	LinkQueue_Show(Q);
}


/**
 * @brief	链式队列初始化
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkQueue_Init(LinkQueue *Q)
{
	Q->front = Q->rear = (QNode *)malloc(sizeof(QNode));

	Q->front->next = NULL;

	return TRUE;
}


/**
 * @brief	链式队列输出
 * @param
 * @note
 * @example
 * @retval
**/
void LinkQueue_Show(LinkQueue Q)
{
	QNode *p = Q.front->next;

	while(p != NULL){
		printf("%d ", p->data);
		p = p->next;
	}
	putchar('\n');
}


/**
 * @brief	链式队列入队
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkQueue_EnQueue(LinkQueue *Q, ElemType elem)
{
	QNode *s = (QNode *)malloc(sizeof(QNode));
	s->next = NULL;
	s->data = elem;

	Q->rear->next = s;		// rear的next指向s
	Q->rear = s;			// rear指向s

	return TRUE;
}


/**
 * @brief	链式队列出队
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkQueue_DeQueue(LinkQueue *Q, ElemType *elem)
{
	if(LinkQueue_IfEmpty(*Q)) return FALSE;
	
	QNode *p = Q->front->next;

	if(p == Q->rear) Q->rear = Q->front;		// 若出对的时最后一个节点, 即rear节点, 出完队要将Q->rear指针修改

	Q->front->next = p->next;
	free(p);

	return TRUE;
}


/**
 * @brief	链式队列判空
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkQueue_IfEmpty(LinkQueue Q)
{
	if(Q.rear == Q.front) return TRUE;

	return FALSE;
}


/**
 * @brief	链式队列的销毁
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkQueue_Destory(LinkQueue *Q)
{
	QNode *p = Q->front->next;
	QNode *temp = p;

	while(p != NULL){
		temp = p;
		p = p->next;
		free(temp);
	}

	Q->rear = Q->front;
	Q->front->next = NULL;

	return TRUE;
}
