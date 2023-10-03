/*混合输入getchar()和scanf*/
#include <stdio.h>

int get_choice(void);
int get_frist();
void count(void);

int main(int argc, char const *argv[])
{
	int choice;
	while((choice = get_choice()) != 'q'){
		switch(choice){
			case 'a': printf("Buy now, sell high\n"); break;
			case 'b': putchar('\a'); break;
			case 'c': count(); break;
			default: printf("Program error\n"); break;
		}
	}

	return 0;
}

int get_choice(void)
{
	int temp;
	printf("Please enter the letter of your choice\n");
	printf("a.adive		b.bell\n");
	printf("c.count 	q.quit\n");
	temp = get_frist();
	while((temp < 'a' || temp > 'c') && temp != 'q'){
		printf("Please enter with a, b, c, q\n");
		temp = get_frist();
	}

	return temp;
}

int get_frist()
{	
	fflush(stdin);
	int temp;
	temp = getchar();
	while(getchar() != '\n'){
		continue;
	}

	return temp;
}

void count(void)
{
	int n , i;
	scanf("%d", &n);
	for(i = 1; i <= n; i ++){
		printf("%d ", i);
	}
	printf("\n");
}