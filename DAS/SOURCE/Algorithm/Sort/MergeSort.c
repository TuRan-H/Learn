/****************************************************************************************************
@author:TuRan
@date:	
@des:	归并排序
****************************************************************************************************/
#include <stdio.h>
#include "..\..\Lib\DAS.h"

typedef struct entry{
	KeyType key;
	DataType data;
}Entry;


typedef struct list{
	int n;
	Entry D[MAXSIZE];
}List;



int main(int argc, char const *argv[])
{


	
	return 0;
}


void InitSort(List *L)
{
	int temp = 0;
	scanf("%d", &temp);
	for(int i = 0; i < MAXSIZE; i ++){
		L->D[i].key = temp;
		L->D[i].data = temp;
		scanf("%d", &temp);
		if(temp == -9999) break;
	}
}

void Merge(List *L, Entry *temp, int low, int n1, int n2)
{
	int i = low, j = low + n1;
	while(i <= low + n1 - 1 && j <= low + n1 + n2 -1){
		if(L->D[i].key <= L->D[j].key){
			*temp ++ = L->D[i];
			i ++;
		}else{
			*temp ++ = L->D[j];
			j ++;
		}
	}
	while(i <= low + n1 - 1){
		*temp ++ = L->D[i];
		i ++;
	}
	while(j <= low + n1 + n2 - 1){
		*temp ++ = L->D[j];
		j ++;
	}
}


void MergeSort(List *L)
{
	Entry temp[MAXSIZE];
	int low, n1, n2, i, size = 1;
	while(size < L->n){
		low = 0;
		while(low + size < L->n){
			n1 = size;
			if(low + size * 2 < L->n)	n2 = size;
			else n2 = L->n - (n1 + size);

			Merge(L, temp + low, low, n1, n2);
			low += n1 + n2;
		}
		for(int i = 0; i < low; i ++){
			L->D[i] = temp[i];
		}
		size *= 2;
	}
}