/****************************************************************************************************
@author: TuRan
@data: 	2022/4/14
@des: 	带头节点的单链表
		链表相关代码
		Linked list code
*****************************************************************************************************/
#include "LinkList.h"


int main(int argc, char *argv[])
{
	LinkList L;
	LNode *p = NULL;


	LNode_Init(&L);

	LNode_Creat_Tail(&L, scanf_string());





	LNode_Show(L);

	LNode_Destory(&L);

	LNode_Show(L);
}


/**
 * @brief	初始化单链表
 * @param	L [LinkList *]
 * @note	LinkList 本身就是 LNode 指针类型, LinkList * 也就是 LNode 的二重指针
 * @retval	bool
*/
bool LNode_Init(LinkList *L)
{
	*L = (LNode *)malloc(sizeof(LNode));	// malloc 返回指针类型(LNode * == LinkList), 
											// 为了能够使得 LinkList 的指针得到改变, 此处必须使用 LinkList *类型
	if(!*L) return FALSE;

	(*L)->next = NULL;						// 初始时将L的next节点设置位null, 防止出现问题

	printf("Initial Success\n");

	return TRUE;
}


/**
 * @brief	输出单链表
 * @param	L [LinkList]
 * @retval	void
*/
void LNode_Show(LinkList L)
{
	printf("Your LinkList is: ");
	LNode *temp;
	temp = L->next;

	while(temp != NULL){
		printf("%d ", temp->data);
		temp = temp->next;
	}
	putchar('\n');
	printf("Show over\n");
}


/**
 * @brief	带头结点的单链表判空操作
 * @param	L [LinkList] 被判断的单链表
 * @retval	bool
**/
bool LNode_IfEmpty(LinkList L)
{
	if(L->next == NULL) return TRUE;
	else return FALSE;
}


/**
 * @brief	求单链表表长
 * @param	L [LinkList] 被求的单链表
 * @retval	单链表的长度 从1开始
**/
int LNode_GetLength(LinkList L)
{
	int i = 1;
	LNode *p = L->next;
	while(p != NULL){
		p = p->next;
		i ++;
	}

	return i - 1;
}


/**
 * @brief	删除 删除单链表 location 位置上的节点
 * @param	L [LinkList *] 操作的单链表
 * @param	location [int] 删除节点的位置，从1开始
 * @param	num [ElemType *] 删除的节点的数据域
 * @retval	bool
*/
bool LNode_Delet(LinkList *L, int location, ElemType *num)
{
	LNode *p = (*L);						// 令 p 指向头节点
	LNode *temp;

	for(int i = 0; i < location - 1; i ++){	// 循环使p指向第 location - 1 个节点
		if(p == NULL) return FALSE;
		p = p->next;
	}

	temp = p->next;							// temp 指向第 location 个节点
	*num = temp->data;						// 返回第 location 节点的数据域
	p->next = p->next->next;				// 修改第 location - 1 节点的next域, 使其指向第 location + 1 个节点
	free(temp);
}


/**
 * @brief	修改 修改单链表某一个位置上的元素
 * @param	L [LinkList]
 * @param	location [int] 该元素在单链表上的位置, 从1开始
 * @param	num [ElemType] 该元素需要修改的内容
 * @retval	bool
*/
bool LNode_Modify(LinkList L, int location, ElemType num)
{
	LNode *p = L->next;						// 令 p 指向 头节点的下一个节点(第1个节点)

	p =  LNode_Find_Node(L, location);


	p->data = num;

	return TRUE;
}


/**
 * @brief	尾插法创建链表
 * @param	L [LinkList *]
 * @param	st_data	[ElemType *] 单链表的内容数组
 * @retval	bool
*/
bool LNode_Creat_Tail(LinkList *L, ElemType *st_data)
{
	if(*L == NULL) return FALSE;
	int i = 0;
	LNode *r = (*L);		// 建立一个指针r, 始终指向最后一个节点


	while(st_data[i] != '\0'){
		LNode *s = (LNode *)malloc(sizeof(LNode));	// 新建一个节点, 用于随后的插入
		s->next = r->next;							// 将新节点的指针指向最后一个指针的next域
		s->data = st_data[i];						// 新节点指针data域赋值
		if(s == NULL) return FALSE;					// 假设新节点创建失败, 返回false
		r->next = s;								// 将最后一个指针的next域指向新节点
		r = s;										// 一切完成, r指向新节点, 也就是当前的最后一个指针
		i ++;
	}
	
	return TRUE;
}


/**
 * @brief	头插法创建单链表
 * @param	L [LinkList *]
 * @param	st_data	[ElemType *] 单链表的内容数组
 * @retval	bool
*/
bool LNode_Creat_Head(LinkList *L, ElemType *st_data)
{
	int i = 0;
	(*L)->next = NULL;
	LNode *s = NULL;


	while(st_data[i] != '\0'){
		s = (LNode *)malloc(sizeof(LNode));		// 新建一个节点, 用于随后的插入
		s->data = st_data[i];
		s->next = (*L)->next;					// 将新节点的next域指向头节点后面的那一个节点
		(*L)->next = s;							// 头节点的next域指向新节点
		i ++;
	}

	return TRUE;
}


/**
 * @brief	插入 在单链表的第location个位置上插入一个节点
 * @param	L [LinkList] 处理的链表
 * @param	location [int] 节点需要插入的位置, location 从1开始(0表示头节点)
 * @param	data [ElemType] 新节点的数据
 * @retval	bool
**/
bool LNode_Insert(LinkList *L, int location, ElemType data)
{
	LNode *p = *L;							// 令p指向头节点L
	int i = 0;								// 计数器i, 表示当前p指向第几个节点(从1开始, 0表示指向头节点)
	while(p != NULL && i < location - 1){	// i从1开始 、 location从1开始。此处进行循环， 让 p 指到 locatoin - 1的位置
		i ++;
		p = p->next;
	}

	LNode *s = (LNode *)malloc(sizeof(LNode));	// 申请节点
	if(!s) return FALSE;
	s->data = data;

	s->next = p->next;							// 重新排列指向关系
	p->next = s;

	return TRUE;
}


/**
 * @brief	后插 在单链表的某一个位置的后面插入一个节点
 * @param	L [LinkList *]
 * @param	location [int] 该节点插入的位置, location 从1开始(0表示头节点)
 * @param	num [ElemType] 该节点data域的数据
 * @note	location 从 0 开始, 0 表示第在头节点后面插入一个节点
 * @retval	bool
*/
bool LNode_Insert_Behind(LinkList *L, int locatoin, ElemType num)
{
	if(locatoin < 0) return FALSE;				// 假设插入的位置小于1, 则插入的位置肯定不存在
	if(locatoin == 0){
		LNode *s = (LNode *)malloc(sizeof(LNode));
		s->next = (*L)->next;
		s->data = num;
		(*L)->next = s;
		return TRUE;
	}
	LNode *s = (LNode *)malloc(sizeof(LNode));	// 新建插入的节点
	LNode *p = (*L)->next;						// 让p节点指向第一个节点(头节点后面的那个节点)
	s->data = num;

	for(int i = 0; i < locatoin - 1; i ++){		// 循环找到location前面一个节点
		if(p == NULL) return FALSE;				// 如何找到p节点指向了最后一个节点的后面一个节点(null), 返回false
		p = p->next;
	}

	s->next = p->next;							// 插入新的节点
	p->next = s;

	return TRUE;
}


/**
 * @brief	前插 在单链表的某一个节点前面插入一个节点
 * @param	L [LinkList *]
 * @param	p [LNode *] 单链表中需要插入节点的地址
 * @param	data [ElemType] 新插入节点data域的数据
 * @note	单链表的前插在物理上没有前插, 只不过是交换了数据域模拟了前插
 * @retval	bool
*/
bool LNode_Insert_Front(LinkList *L, LNode *p, ElemType data)
{
	int temp = 0;

	LNode *s = (LNode *)malloc(sizeof(LNode));	// 申请新的节点
	s->data = data;
	s->next = p->next;	// 链接节点
	p->next = s;

	temp = p->data;		// 交换新的节点和被插入节点的数据以实现"假前插"
	p->data = s->data;
	s->data = temp;

	return TRUE;
}


/**
 * @brief	按位查找 查找单链表中的某一个元素, 根据位置查找
 * @param	L [LinkList] 被查找的单链表
 * @param	location [int] 该元素在单链表上的位置 (从1开始)
 * @retval	ElemType
*/
ElemType LNode_Find_Location(LinkList L, int location)
{
	LNode *p = L->next;		// 令 p 指向头结点的后面一个节点
	for(int i = 0; i < location - 1; i ++){	// 循环, 使p指向第location个节点
		if(p == NULL) return ERROR;
		p = p->next;
	}

	return p->data;			// 返回第location个节点的数据域
}


/**
 * @brief	按值查找 查找单链表中的某一个元素, 根据元素类型查找
 * @param	L [LinkList]
 * @note	elem [ElemType]
 * @retval	返回元素在单链表中的位置, 从1开始
*/
int LNode_Find_Elem(LinkList L, ElemType elem)
{
	LNode *p = L->next;			// 令p指向头结点的后面一个节点
	int i = 1;
	while(p != NULL){			// 循环, 使p->data和每个节点的data域作比较
		if(p->data == elem) return i;	// 若相同, 则返回节点的位置
		p = p->next;
		i ++;
	}

	return ERROR;				// 若不同, 则返回-9999
}


/**
 * @brief	查找 找到第 location 个节点, 并返回其节点的地址
 * @param	L [LinkList]
 * @param	location [int] 节点的位置, 从1开始(0表示头节点)
 * @retval	第 location 个节点的地址
**/
LNode *LNode_Find_Node(LinkList L, int location)
{
	LNode *p = L;
	for(int i = 0; i < location; i ++){
		p = p->next;
	}

	return p;
}


/**
 * @brief	双链表的摧毁
 * @param	L [LinkList *] 执行摧毁的链表
 * @retval	bool
**/
bool LNode_Destory(LinkList *L)
{
	if((*L) == NULL) return FALSE;

	LNode *p = (*L)->next;
	LNode *temp = p;
	while(p != NULL){
		temp = p;
		p = p->next;
		free(temp);
	}

	return TRUE;
}