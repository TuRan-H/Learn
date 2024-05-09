#ifndef __SQSTACK_H__
#define __SQSTACK_H__
#include <stdio.h>
#include <stdlib.h>
#include "../../lib/DAS.h"

typedef struct Sequence_Stack{
	ElemType *data;
	int top;					// top用来指向栈顶元素
}SqStack;


bool SqStack_Init(SqStack *L);
bool SqStack_IfEmpty(SqStack L);
void SqStack_Show(SqStack L);
bool SqStack_PUSH(SqStack *L, ElemType elem);
bool SqStack_POP(SqStack *L, ElemType *elem);
bool SqStack_GetTop(SqStack L, ElemType *elem);
void SqStack_Destory(SqStack *L);


#endif