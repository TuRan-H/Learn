/****************************************************************************************************
@author: TuRan
@date:	2022/6/15
@des:	顺序栈相关代码
		top指向栈顶元素
****************************************************************************************************/
#include "SqStack.h"


int main(int argc, char const *argv[])
{
	SqStack L;

	ElemType temp;

	SqStack_Init(&L);

	SqStack_PUSH(&L, 3);
	SqStack_PUSH(&L, 3);
	SqStack_POP(&L, &temp);
	SqStack_POP(&L, &temp);
	SqStack_PUSH(&L, 3);
	SqStack_PUSH(&L, 3);
	SqStack_PUSH(&L, 9);

	SqStack_GetTop(L, &temp);

	printf("%d\n", temp);


	SqStack_Show(L);


	return 0;
}


/**
 * @brief	顺序栈初始化函数
 * @param	L [SqStack *]
 * @retval	bool
**/
bool SqStack_Init(SqStack *L)
{
	L->top = -1;				// 初始时让top指向-1, 表示当前栈中无任何元素
	L->data = (ElemType *)malloc(sizeof(ElemType) * INITSIZE);

	if(!L->data) return FALSE;

	return TRUE;
}


/**
 * @brief	顺序队列遍历并输出
 * @param
 * @note
 * @example
 * @retval
**/
void SqStack_Show(SqStack L)
{
	for(int i = 0; i < L.top + 1; i ++){
		printf("%d ", L.data[i]);
	}

	putchar('\n');
}


/**
 * @brief	顺序栈判断栈空函数
 * @param
 * @note
 * @example
 * @retval	TRUE -- 站控 || FALSE -- 栈非空
**/
bool SqStack_IfEmpty(SqStack L)
{
	if(L.top == -1) return TRUE;
	else return FALSE;
}


/**
 * @brief	顺序栈判断栈满函数
 * @param	
 * @note
 * @example
 * @retval	TRUE -- 栈满 || FALSE -- 栈非满
**/
bool SqStack_IfFull(SqStack L)
{
	if(L.top == MAXSIZE - 1) return TRUE;
	else return FALSE;
}


/**
 * @brief	顺序栈出栈函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqStack_POP(SqStack *L, ElemType *elem)
{
	if(SqStack_IfEmpty(*L)) return FALSE;
	
	*elem = L->data[L->top];
	L->top --;

	return TRUE;
}


/**
 * @brief	顺序栈入站函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqStack_PUSH(SqStack *L, ElemType elem)
{
	if(SqStack_IfFull(*L)) return FALSE;

	L->top ++;
	L->data[L->top] = elem;

	return TRUE;
}


/**
 * @brief	顺序栈取栈顶元素函数
 * @param
 * @note
 * @example
 * @retval
**/
bool SqStack_GetTop(SqStack L, ElemType *elem)
{
	if(SqStack_IfEmpty(L)) return FALSE;

	*elem = L.data[L.top];

	return TRUE;
}


/**
 * @brief	顺序栈的摧毁
 * @param
 * @note
 * @example
 * @retval
**/
void SqStack_Destory(SqStack *L)
{
	L->top = -1;
	free(L->data);
}