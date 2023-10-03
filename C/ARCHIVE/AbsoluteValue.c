/*  输入一个数,将其取绝对值  */
#include <stdio.h>
int main(int argc, char const *argv[])
{
	int input;
	scanf("%d", &input);
	//如果作为调用函数的话,这玩意就是参数

	if(input >= 0){
		input = input;
	}
	else{
		input = -input;
	}

	printf("%d\n", input);

	return 0;
}