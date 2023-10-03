/*
	使用了经典的是十字链表法， 用来压缩稀疏矩阵。
	十字链表法有向右和向下两种方式， 本例中使用的是向右
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define TRUE 1
#define FALSE 0
#define MAX 20

struct Node{
	int from, to;
	struct Node *in_pointer;
	struct Node *out_pointer;
};

struct Cross_List{
	int vex, edge;
	struct{
		char head_else;
		struct Node *in_pointer;
		struct Node *out_pointer;
	}head[MAX];
};

void creat(struct Cross_List *map);
void show(struct Cross_List *map);
void insert(struct Cross_List *map, char from, char to);


void creat(struct Cross_List *map)
{
	char from, to;
	printf("要创建几个节点， 几条边: ");
	scanf("%d%d", &map->vex, &map->edge);
	getchar();
	for(int i = 0; i < map->vex; i ++){
		map->head[i].head_else = i + 'A';
		map->head[i].in_pointer = NULL;
		map->head[i].out_pointer = NULL;
	}
	printf("请输入图的边: \n");
	for(int i = 0; i < map->edge; i ++){
		scanf("%c%c", &from, &to);
		getchar();
		insert(map, from, to);
	}
}


void insert(struct Cross_List *map, char from, char to)
{
	struct Node *p = malloc(sizeof(struct Node));
	struct Node *now;
	struct Node *before;
	struct Node *q;
	p->from = from - 'A';
	p->to = to - 'A';
}
