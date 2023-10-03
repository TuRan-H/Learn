/****************************************************************************************************
@author: TuRan
@data: 2022/4/10
@des: 	线性表相关代码
		Sequence list code
****************************************************************************************************/
#include "SqList.h"


/**
 * @brief	初始化线性表
 * @param	L [SqList *] 线性表的地址
 * 
*/
bool SqList_Init(SqList *L)
{
	(*L).data = (ElemType *)malloc(sizeof(ElemType) * INITSIZE);		// 给data malloc数据空间

	if(!L->data) return FALSE;											// 假设L->data不存在, 返回FALSE

	(*L).length = 0;													// 将初始时的length设置为0
	(*L).maxsize = INITSIZE;											// 将初始时的 maxsize 设置为 INITSIZE

	printf("SqList Init Success\n");

	return TRUE;
}


/**
 * @brief	遍历链表
 * @param	L [SqList *] 线性表的指针
 * @retval	void
*/
void SqList_Show(SqList L)
{
	int i;
	printf("Your Sqlist is:\n");
	for(i = 0; i < L.length; i ++){
		printf("%d ", L.data[i]);
	}

	putchar('\n');
	putchar('\n');
}


/**
 * @brief	顺序表判空操作
 * @param	L [SqList] 被查找的单链表
 * @retval	bool
**/
bool SqList_IfEmpty(SqList L)
{
	if(L.length == 0) return TRUE;
	else return FALSE;
}


/**
 * @brief
 * @param
 * @note
 * @example
 * @retval
**/
int SqList_GetLength(SqList L)
{
	int temp = L.length;

	return temp;
}


/**
 * @brief	创建线性表
 * @param	L [SqList *] 线性表的头指针
*/
bool SqList_Creat(SqList *L, ElemType *st)
{
	int i = 0;

	while(st[i] != '\0'){
		if(L->length + 1 > L->maxsize){		// 假设当前线性表的长度大于最大长度, 则扩大线性表
			SqList_Extend(L);
		}
		L->data[i] = st[i];
		i ++;
		L->length ++;
	}
	return TRUE;
}


/**
 * @brief	延长链表
 * @param	L [SqList *] 线性表的指针
 * @note 	重新创建一个数据域, 该数据域的长度为原来数据域的最大长度 + INCSIZE
 * 			之后将原来的数据域的地址设置为新的数据域
 * 			并且将原来的数据域free了
*/
bool SqList_Extend(SqList *L)
{
	ElemType *data;
	data = (ElemType *)malloc(sizeof(ElemType) * L->maxsize + INCSIZE);		// 新建一个数据域
	if(!data) return FALSE;													// 假设数据域创建失败, 返回false

	for(int i = 0; i < L->length; i ++) data[i] = L->data[i];				// 挨个将数据传到新的数据域中


	free(L->data);															// 释放原先的链表的数据域

	// 将原先的链表的指针域以及各个域全部都设置为新的
	L->data = data;
	L->maxsize = L->maxsize + INCSIZE;


	return TRUE;
}


/**
 * @brief	(增)在线性表中某一个位置后面插入某一个元素
 * @param	L [SqList *] 线性表的指针
 * @param	location [int] 插入的位置, Location 从 1 开始计数
 * @param	data [ElemType] 所需要插入的数据
 * @note	location是从1开始计数的, 因此location作为插入位置比实际上的插入位置location' 多了一个位
 * @retval	int
*/
bool SqList_Insert(SqList *L, int location, ElemType data)
{
	if(L->length + 1 > L->maxsize){
		SqList_Extend(L);
	}

	for(int i = L->length; i >= location; i --){		// 将 L->length 一直到 location(location' + 1)全部都向后移一位
		L->data[i] = L->data[i - 1];
	}

	L->data[location] = data;							// 将data插入在 location' + 1 的位置上
	
	L->length ++;										// L->length 加一

	return TRUE;	
}


/**
 * @brief	(删)在线性表某一位位置上删除一个元素, 并且返回元素内
 * @param	L [SqList *] 线性表的指针
 * @param	location [int] 需要删除的位置
 * @param	data [ElemType *] 删除的元素的内容
 * @retval	int
*/
bool SqList_Delet(SqList *L, int location, ElemType *data)
{
	if(location > L->length || L->length == 0) return FALSE;		// 假设location 比 L的长度大, 或者L的长度本身就位0,  返回false

	*data  = L->data[location - 1];									// 将data传过去

	for(int i = location - 1; i < L->length; i ++){					// 从location' 处开始, 所有数据一次向左移一位
		L->data[i] = L->data[i + 1];
	}

	L->length --;

	return TRUE;
}


/**
 * @brief	(改)修改某一个位置上面的元素
 * @param	L [SqList *] 线性表的指针
 * @param	location [int] 需要修改的元素所在的位置
 * @param	data [ElemType] 需要修改的元素的内容
 * @retval	int
*/
bool SqList_Modify(SqList *L, int location, ElemType data)
{
	if(location > L->length) return FALSE;

	L->data[location - 1] = data;

	return TRUE;
}


/**
 * @brief	(查)查找某一个位置上面的元素并且返回
 * @param	L [SqList *] 线性表的指针
 * @param	location [int] 元素的位置
 * @param	data [ElemType *] 返回的元素
 * @retval	int
*/
bool SqList_Find_Location(SqList *L, int location, ElemType *data)
{
	if(location > L->length) return FALSE;

	*data = L->data[location - 1];

	return TRUE;
}


/**
 * @brief	按值查找一个元素
 * @param	L [SqList *] 查找的单链表对象
 * @param	elem [int] 查找的元素
 * @retval	假设查找到了某一个元素, 返回其位序(从1开始)
 * 			假设没有查找到元素, 返回一个FALSE
*/
bool SqList_Find_Elem(SqList *L, int elem)
{
	for(int i = 0; i < L->length; i ++){
		if(L->data[i] == elem)
			return i + 1;
	}

	return FALSE;
}


/**
 * @brief	销毁线性表
 * @param	L [SqList *]
 * @retval	int
*/
bool SqList_Destory(SqList *L)
{
	free(L->data);
	L->length = 0;
	L->maxsize = 0;

	printf("Destory Success\n");

	return TRUE;
}

//---------------------------------------------------------------------------

bool function(SqList *L)
{
	int  k = 0;

	for(int i = 1; i < L->length; i ++){
		if(L->data[i - 1] == L->data[i]) k ++;
		else L->data[i - k] = L->data[i];
	}

	L->length -= k;

	return TRUE;
}

bool function2(SqList  *L)
{
	if (L->length == 0) return FALSE;
	int i = 0, j = 0;
	
	for(i = 0, j = 1; j < L->length; j ++){
		if(L->data[j] != L->data[i]){
			i ++;
			L->data[i] = L->data[j];
		}
	}
	L->length = i + 1;
	
	return TRUE;
}


SqList function3(SqList *p, SqList*q)
{
	SqList L;
	L.data = (ElemType *)malloc(sizeof(ElemType) * (q->length + p->length));

	int i = 0, j = 0,k = 0;
	for(i = 0, j = 0; i < p->length && j < q->length; ){
		if(p->data[i] <= q->data[j])	L.data[k++] = p->data[i++];
		else if(p->data[i] > q->data[j])	L.data[k++] = q->data[j++];
	}
	for( ; i < p->length; i ++){
		L.data[++k] = L.data[i];
	}
	for( ; j < q->length; j ++){
		L.data[++k] = L.data[j];
	}

	L.length = p->length + q->length;

	return L;
}



// 1 2 2 2 2 3 3 3 4 4 5 
// ----------------------------------------------------------------------------
int main(int argc, char const *argv[])
{
	SqList L;

	SqList p, q;
	SqList_Creat(&p, scanf_string());
	SqList_Creat(&q, scanf_string());

	
	L = function3(&p, &q);

	SqList_Show(L);








	SqList_Show(L);


	return 0;
}
