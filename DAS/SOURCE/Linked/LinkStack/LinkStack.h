#ifndef __LINKSTACK_H__
#define __LINKSTACK_H__
#include <stdio.h>
#include <stdlib.h>
#include "../../lib/DAS.h"


typedef struct Linked_Stack{
	struct Linked_Stack *next;
	ElemType data;
}SNode, *LinkStack;


bool LinkStack_PUSH(LinkStack *S, ElemType elem);
bool LinkStack_POP(LinkStack *S, ElemType *elem);
void LinkStack_Show(LinkStack S);
bool LinkStack_Init(LinkStack *S);
bool LinkStack_IfEmpty(LinkStack S);
bool LinkStack_GetTop(LinkStack S, ElemType *elem);
bool LinkStack_Destory(LinkStack *S);


#endif