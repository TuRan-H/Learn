/****************************************************************************************************
@author:TuRan
@data:	
@des:	文件的输出输出, 第三节: 怎样向文件读写一个字符串 相关例题
****************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void question0();
void test();


int main()
{
	test();
}

void test()
{
	FILE *fp;
	fp = fopen("data.dat", "rb+");
	int a = 3;
	int b = 5;

	fwrite(&a, sizeof(int), 1, fp);
	fread(&b, sizeof(int), 1, fp);
	printf("%d", b);
}

/**
 * @brief	从键盘中读入若干字符, 对他们按照字典顺序排列并输出到磁盘文件中保存
*/
void question0()
{
	const int m = 3, n = 10;
	FILE *fp = fopen("data.dat", "w");	// 以写入方式打开文件
	char str[m][n], temp[n];

	printf("enter string\n");
	for(int i = 0; i < m; i ++) gets(str[i]);

	for(int i = 0, k = 0; i < m-1; i ++){
		k = i;
		for(int j = i+1; j < m; j ++) if((strcmp(str[k], str[j])) > 0) k = j;
		if(k != i){
			strcpy(temp, str[k]);
			strcpy(str[k], str[i]);
			strcpy(str[i], temp);
		}
	}

	for(int i = 0; i < m; i ++){
		fputs(str[i], fp);
		fputc('\n', fp);
	} 
}

void question1()
{
	const int SIZE = 10;
	struct Student_Type{
		char name[10];
		int num;
		int age;
		char addr[15];
	}stud[SIZE], temp;

	stud[0].num = 1;
	FILE *fp1, *fp2;
	fp1 = fopen("data.dat", "wb");
	fwrite(&stud[0], sizeof(struct Student_Type), 1, fp1);
	fp2 = fopen("data.dat", "rb");
	fread(&temp, sizeof(struct Student_Type), 1, fp2);
	printf("%d", temp.num);
}