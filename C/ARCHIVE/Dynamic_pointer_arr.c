/*
malloc 函数的演示程序,创建了一个指针 ptd ,指针一开始没有给予他空间,之后使用了 malloc 函数赋予指针空间
空间赋予完毕之后,输入多个 double 型整数并打印值至 stdout
*/

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	double *ptd;
	int max;
	int number;
	int i = 0;

	puts("What is th maxinum number of type double entries?");
	if(scanf("%d", &max) != 1){									//防止错误输入
		puts("Number not corretcly entered --bye.");
		exit(EXIT_FAILURE);
	}
	ptd=(double*) malloc(max * sizeof(double));					//为 ptd 指针创建数据空间
	if(ptd == NULL){
		puts("Memory allocation failed. Goodbye.");
		exit(EXIT_FAILURE);
	}															//若数据空间内存创建失败则返回 EXIT_FAILURE


	puts("Enter the values (q to quit)");
	while(i < max && scanf("%lf", &ptd[i]) == 1){
		i ++;
	}
	printf("Here are your %d entries:\n", number = i);
	for(i = 0; i < number; i ++){
		printf("%7.3f ", ptd[i]);
		if(i % 7 == 6)
			putchar('\n');
	}
	if(i % 7 != 0){
		putchar('\n');
	}


	puts("Done");
	free(ptd);													//释放空间




	return 0;
}