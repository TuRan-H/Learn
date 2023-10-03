/*判断长度,输入一个整型数,根据输入的值判断该数的大小是多少位数*/
#include <stdio.h>

int estimate_length(int input);
int absolute_value(int input);

int main(int argc, char const *argv[])
{
	int i;
	scanf("%d", &i);

	i = estimate_length(i);
	printf("%d\n", i);
}

int estimate_length(int input)
{
	input = absolute_value(input);

	int i, length = 1;
	i = 10;
	if(input > 10){
		while(input / i != 0){
			length ++;
			i *= 10;
		}
	}else if(input <= 10 && input != 0){
		length = 1;
	}else{
		length = 0;
	}
	//判断整型数的长度

	return length;

}

int absolute_value(int input)
{
	// int input;
	// scanf("%d", &input);
	//如果作为调用函数的话,这玩意就是参数

	if(input >= 0){
		input = input;
	}
	else{
		input = -input;
	}

	// printf("%d\n", input);

	return input;
}