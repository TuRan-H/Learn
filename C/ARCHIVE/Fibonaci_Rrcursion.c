#include <stdio.h>

int Fibonaci(int n);

int main(int argc, char const *argv[])
{
	
	int x = Fibonaci(8);
	printf("%d\n", x);


	return 0;
}

int Fibonaci(int n)
{
	if(n == 0)
		return 0;
	else if(n == 1)
		return 1;
	else
		return Fibonaci(n - 1) + Fibonaci(n - 2);
}
