/*
创建了一个结构体,该结构体的内容有两个是指针类型的 char ,因为是指针类型的并且没有初始化,所以不能直接往里面赋值
因此我使用了 malloc 函数给结构体的两个指针类型的 char 创建内存空间然后设置了一个中间变量 temp[] 用来储存 stdin
的输入,然后使用 StrCopy 函数将 temp 的值拷贝到 两个指针类型的 char 里面(实际上指针类型的 char 就是 char 类型的数组)
malloc 的过程中使用了 strlen 函数来计算 malloc 
所需要的空间
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "D:\Project\C\lib\lib_c.h"

#define SLEN 81

struct namect{
	char *fname;
	char *lname;
	int letters;
};

void getinfo(struct namect *pst);
void makeinfo(struct namect *pst);
void showinfo(struct namect *pst);
void cleanup(struct namect *pst);

int main(int argc, char const *argv[])
{
	struct namect person;

	getinfo(&person);
	makeinfo(&person);
	showinfo(&person);
	cleanup(&person);


	return 0;
}


void getinfo(struct namect *pst)
{
	char temp[SLEN];
	printf("Pleas enter your frist name\n");				
	s_gets(temp, SLEN);									//因为定义的 struct 的 fname 是一个未初始化的指针,因此不能随随便便的对 pst->fname 赋值
	pst->fname = (char *) malloc(strlen(temp) + 1);
	StrCopy(pst->fname, temp);							//使用 StrCopy 函数把 temp 的值拷贝至 pst->fname 中
	printf("Please enter your last name\n");
	s_gets(temp, SLEN);
	pst->lname = (char *) malloc(strlen(temp) + 1);
	StrCopy(pst->lname, temp);
}

void makeinfo(struct namect *pst)
{
	pst->letters = strlen(pst->fname) + strlen(pst->lname);
}

void showinfo(struct namect *pst)
{
	printf("%s %s, Your name contain %d letters \n", pst->fname, pst->lname, pst->letters);
}

void cleanup(struct namect *pst)
{
	free(pst->fname);
	free(pst->lname);
}