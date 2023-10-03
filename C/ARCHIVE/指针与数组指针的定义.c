#include <stdio.h>

int main(int argc, char const *argv[])
{
	char arr[80] = "test";
	int i = 3;
	int *p = &i;

	// puts(arr);

	printf("%d\n", *arr);			//arr是一个指针,指向的数组的首元素,因此*arr也就是该数组首元素
	printf("%d\n", arr[0]);			//arr[0]与*arr是一个概念
	printf("%s\n", arr);			//arr是一个地址,%s需要的也是一个char *,printf可以把这个地址的字符串打印出来(一直到'\0')
	printf("%p\n", arr);			//数组可以使用数组名本身来代表一个地址,也可以加上&来代表一个地址,二者是一个意思
	printf("%p\n", &arr);			//数组是特殊情况,数组本身就是一个指针,指向数组的首元素
	printf("%p\n", &arr[2]);
	printf("%c\n", 97);

	puts("这里是普通指针\n");

	printf("%p\n", p);				//指针本身代表一个地址,该地址是指向的元素的地址
	printf("%p\n", &p);				//&指针,表示指针的地址
	printf("%d\n", *p);


	return 0;
}