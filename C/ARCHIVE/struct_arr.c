/*
结构体数组示例程序,
定义了一个结构体叫 books
声明了一个结构体数组叫 librarys[MAX]
使用该结构体存储了多本书的书名作者价格
*/

#include <stdio.h>
#include <stdlib.h>
#include "D:\Work\Storage\C-CPP\Lib\lib_c.h"

#define MAXTITL 40
#define MAXAUTL 40
#define MAXBKS 100

struct book{
	 char title[MAXTITL];
	 char author[MAXAUTL];
	 float value;
};

int main(int argc, char const *argv[])
{
	struct book librarys[MAXBKS];
	int count = 0;
	int index;

	printf("Please enter the book title\n");
	printf("Press [enter] at the start of a line to stop\n");

	while(count < MAXBKS && s_gets(librarys[count].title, MAXTITL) != NULL && librarys[count].title[0] != '\0'){
		printf("Now enter the author\n");
		s_gets(librarys[count].author, MAXAUTL);
		printf("Now enter the value\n");
		scanf("%f",&librarys[count ++].value);
		while(getchar() != '\n'){
			continue;
		}
		if(count < MAXBKS){
			printf("Enter the next title\n");
		}
	}
	if(count > 0){
		printf("Here is the list of your books\n");
		for(index = 0; index < count; index ++){
			printf("%s by %s :$%.2f \n", librarys[index].title, librarys[index].author, librarys[index].value);
		}
	}else
		printf("No books\n");



	return 0;
}