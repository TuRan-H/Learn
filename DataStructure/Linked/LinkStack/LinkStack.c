/****************************************************************************************************
@author:TuRan
@date:	2022/6/16
@des:	带头结点的链栈.
		出栈和入栈都在头节点后面进行
****************************************************************************************************/
#include "LinkStack.h"

int main(int argc, char const *argv[])
{
	LinkStack S;
	ElemType temp;

	LinkStack_Init(&S);


	LinkStack_PUSH(&S, 3);
	LinkStack_PUSH(&S, 3);
	LinkStack_PUSH(&S, 4);
	LinkStack_POP(&S, &temp);

	printf("TOP is %d\n", temp);
	LinkStack_Destory(&S);

	LinkStack_Show(S);



	return 0;
}


/**
 * @brief	链栈的初始化
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkStack_Init(LinkStack *S)
{
	(*S) = (SNode *)malloc(sizeof(SNode));
	if(!S) return FALSE;

	(*S)->next = NULL;

	return TRUE;
}


/**
 * @brief	链栈判断栈空函数
 * @param
 * @note
 * @example
 * @retval	TRUE--栈空 || FALS--栈非空
**/
bool LinkStack_IfEmpty(LinkStack S)
{
	if(S->next == NULL) return TRUE;

	return FALSE;
}


/**
 * @brief	链栈的循环遍历
 * @param
 * @note
 * @example
 * @retval
**/
void LinkStack_Show(LinkStack S)
{
	SNode *p = S->next;

	while(p != NULL){
		printf("%d ", p->data);
		putchar('\n');

		p = p->next;
	}
}


/**
 * @brief	链栈的入栈函数
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkStack_PUSH(LinkStack *S, ElemType elem)
{
	SNode *p = (*S)->next;
	SNode *s = (SNode *)malloc(sizeof(SNode));
	s->data = elem;

	s->next = p;
	(*S)->next = s;

	return TRUE;
}


/**
 * @brief	链栈的出栈函数
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkStack_POP(LinkStack *S, ElemType *elem)
{
	if(LinkStack_IfEmpty(*S)) return FALSE;
	
	SNode *p = (*S)->next;
	*elem =  p->data;

	(*S)->next = p->next;
	free(p);

	return TRUE;
}


/**
 * @brief	链栈的取头节点
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkStack_GetTop(LinkStack S, ElemType *elem)
{
	if(LinkStack_IfEmpty(S)) return FALSE;


	SNode *p = S->next;

	*elem = p->data;

	return TRUE;
}


/**
 * @brief	链栈的销毁代码
 * @param
 * @note
 * @example
 * @retval
**/
bool LinkStack_Destory(LinkStack *S)
{
	if(LinkStack_IfEmpty(*S)) return FALSE;

	SNode *p = (*S)->next;
	SNode *temp = p;

	while(p != NULL){
		temp = p;
		p = p->next;
		free(temp);
	}

	return TRUE;
}