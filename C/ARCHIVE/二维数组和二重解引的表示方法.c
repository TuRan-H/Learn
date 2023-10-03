#include <stdio.h>



int main(int argc, char const *argv[])
{
	int zippo[4][2] = {
		{1,2},
		{3,4},
		{4,6},
		{7,8},
	};
	int *i = 0, j = 0, k = 0;
	int (*point)[2];



	printf("%p\n", zippo);				//zippo表示一个地址
	printf("%p\n", zippo[0]); 			//zippo[0]表示一个地址
	printf("%p\n", &zippo[0]); 			//&zippo[0]也表示一个地址
	printf("%d\n", zippo[0][0]);		//zippo[0][0]表示一个数据
	printf("%p\n", &zippo[0][0]);		//&zippo[0][0]表示一个地址

	printf("\n");
	printf("\n");

	printf("%d\n", *zippo[1]);
	printf("%p\n", zippo[1]);
	printf("%p\n", &zippo[1][0]);

	printf("\n");
	printf("\n");

	printf("%p\n", zippo + 1);
	printf("%p\n", zippo[0] + 1);


	printf("\n");
	printf("\n");


	i = *zippo; 						//zippo是一个双级指针
	printf("%p\n", i);					//zippo指向的是zippo[0],但是zippo[0]是一个地址,zippo[0]指向的是zippo[0][0]
	j = **zippo;						//使用两次解引用就可以的到zippo[0][0]的值
	printf("%d\n", j);
	point = zippo;
	printf("%p\n", point);					//point是一个内含两个int类型地址值的指针,(*point)的用意是使用[]的优先级高于*,使用括号来给point优先定义指针

	printf("\n");
	printf("\n");


	k = *(*(zippo + 2) + 1);
	printf("%d\n", k);

	return 0;
};