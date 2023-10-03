/*
 * @lc app=leetcode.cn id=1 lang=c
 *
 * [1] 两数之和
 */

// @lc code=start
#include <stdio.h>
#include <stdlib.h>


// typedef int ElemType;
// typedef struct Sequence_Stack{
// 	ElemType *data;

// }SqStack;



/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* twoSum(int* nums, int numsSize, int target, int* returnSize)
{
	int i = 0, j = 0;
	int *retarr = (int *)malloc(sizeof(int) * 2);


	for(i = 0; i < numsSize; i ++){
		for(j = i + 1; j < numsSize; j ++){
			if(nums[i] + nums[j] == target){
				retarr[0] = i;
				retarr[1] = j;
				*returnSize = 2;
				return retarr;
			}
		}
	}

	*returnSize = 0;
	return NULL;

}

// @lc code=end

