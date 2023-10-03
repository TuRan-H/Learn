/*递归实现二进制数转十进制数*/
#include <stdio.h>

void BinaryChange(unsigned long n);

int main(int argc, char const *argv[])
{
	unsigned long number;
	scanf("%ld", &number);
	BinaryChange(number);
}

void BinaryChange(unsigned long n)
{
	int r = n % 2;
	if(n >= 2){
		BinaryChange(n / 2);
	}
	putchar(r == 0? '0' : '1');
}