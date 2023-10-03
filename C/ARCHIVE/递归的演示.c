/*递归的演示*/
#include <stdio.h>

void Up_Down(int);

int main(int argc, char const *argv[])
{
	Up_Down(1);

	return 0;
}

void Up_Down(int n)
{
	printf("Level %d: n location %p\n", n, &n);
	if(n < 4)
		Up_Down(n + 1);
	printf("LEVEL %d: n location %p\n", n, &n);
}


