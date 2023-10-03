#ifndef __SQLIST_H__
#define __SQLIST_H__
#include "../../Lib/DAS.h"


// 线性表结构体
typedef struct Sequence_List{
	ElemType *data;		// 数据域
	int maxsize;		// 最大长度
	int length;			// 已占用长度
}SqList;


bool SqList_Init(SqList *L);
bool SqList_Creat(SqList *L, ElemType *st);
bool SqList_Destory(SqList *L);
void SqList_Show(SqList L);
bool SqList_Extend(SqList *L);
bool SqList_Insert(SqList *L, int location, ElemType data);
bool SqList_Delet(SqList *L, int location, ElemType *data);
bool SqList_Modify(SqList *L, int location, ElemType data);
bool SqList_Find_Location(SqList *L, int location, ElemType *data);
bool SqList_Find_Elem(SqList *L, int elem);
bool SqList_IfEmpty(SqList L);
int SqList_GetLength(SqList L);


#endif