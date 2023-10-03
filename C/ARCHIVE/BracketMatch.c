/*
	括号匹配程序
*/
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0
#define MAXSIZE 20
#define INITSIZE 10

typedef struct{
	char *data;
	int top;
}SqStack;
typedef struct{
	char *data;
	int maxsize;			
	int length;				
}SqList;

bool InitStack(SqStack *S);
bool IfEmpty(SqStack S);
bool push(SqStack *S, char e);
bool ShowList(SqStack S);
bool BracketMatch(char str[], int length);
bool pop(SqStack *S, char *e);
bool InitList(SqList *L);
bool DestroyList(SqList *L);
bool CreatList(SqList *L);
void ShowList(SqList L);

int main(int argc, char const *argv[])
{
	SqList L;
	InitList(&L);
	printf("Please enter your char\n");
	CreatList(&L);
	ShowList(L);




	// int length = 10;
	// char str[10] = "()(())(())";

	// int i = BracketMatch(str, length);

	
	// if(i == 1)
	// 	printf("TRUE\n");	
	// else
	// 	printf("FALSE\n");
	return 0;
}

bool IncreaseSize(SqList *L, int len)
{
	int i = 0;													// 计数器
	char *p = (*L).data;
	(*L).data = (char *) malloc(sizeof(char) * (INITSIZE + len));	//注意,这里是直接你妈的新创建了一个数据空间!!!
	for(i = 0; i < ((*L).maxsize + len); i ++){
		(*L).data[i] = 0;
	}
	for(i = 0; i < L->maxsize; i ++){							//这个for函数把原来数据空间的值转移到了新的数据空间
		(*L).data[i] = p[i];
	}

	(*L).maxsize = (*L).maxsize + len;

	free(p);													//释放原来内存


	return TRUE;;
}


void ShowList(SqList L)
{
	int i;
	for(i = 0; i < L.length; i ++){
		printf("%c ", L.data[i]);
	}

	putchar('\n');
}

bool DestroyList(SqList *L)
{
	free(L->data);
	L->length = 0;
	L->maxsize = 0;
	return TRUE;
}


bool CreatList(SqList *L)
{
	int i = 0;
	char num;
	int increasenum = 5;
	scanf("%c", &num);
	while(num != '!'){
		if(L->length + 1 > L->maxsize){
			IncreaseSize(L, increasenum);
		}
		L->data[i] = num;
		i ++;
		L->length ++;
		scanf("%c", &num);
	}

	return TRUE;

}

bool InitList(SqList *L)
{
	(*L).data = (char *) malloc(sizeof(char) * INITSIZE);
	if(!L->data){
		return FALSE;
	}
	(*L).length = 0;
	(*L).maxsize = INITSIZE;

	return TRUE;;
}




/*
	length表示char数组的长度
	str表示工作数组
*/
bool BracketMatch(SqList L)
{
	SqStack S;
	InitStack(&S);
	while(){
		if(str[i] == '(' || str[i] == '[' || str[i] == '{')
			push(&S, str[i]);
		else{
			if(IfEmpty(S))
				return FALSE;
			else{
				char temp;
				pop(&S, &temp);
				if(str[i] == ')' && temp != '(')
					return FALSE;
				if(str[i] == ']' && temp != '[')
					return FALSE;
				if(str[i] == '}' && temp != '{')
					return FALSE;
			}
		}
	}
	if(IfEmpty(S))
		return TRUE;
	else
		return FALSE;
}

bool InitStack(SqStack *S)
{
	S->data = (char *) malloc(sizeof(char) * MAXSIZE);
	if(S->data == NULL)
		return FALSE;
	S->top = -1;

	return TRUE;
}

bool IfEmpty(SqStack S)
{
	if(S.top == -1)
		return TRUE;
	else
		return FALSE;
}

bool pop(SqStack *S, char *e)
{
	if(S->top == -1)
		return FALSE;
	*e = S->data[S->top --];

	return TRUE;
}

bool push(SqStack *S, char e)
{
	if(S->top == MAXSIZE - 1){
		return FALSE;
	}
	S->top ++;
	S->data[S->top] = e;	

	return TRUE;
}

bool ShowList(SqStack S)
{
	if(S.top == -1){
		return FALSE;
	}
	for(int i = S.top; i >= 0; i --){
		printf("%c ", S.data[i]);
	}	

	return TRUE;
}

