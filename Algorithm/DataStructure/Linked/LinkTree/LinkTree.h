/****************************************************************************************************
@author:TuRan
@date:	2022/6/28
@des:	二叉树头文件
****************************************************************************************************/
#ifndef __LINKTREE_H__
#define __LINKTREE_H__
#include "../../lib/DAS.h"

typedef struct Linked_Binary_Tree{
	ElemType data;
	struct Linked_Binary_Tree *lchild, *rchlid;
}BiTNode, *BiTree;



#endif