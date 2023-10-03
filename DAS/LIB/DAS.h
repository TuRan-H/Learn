/**
 *                             _ooOoo_
 *                            o8888888o
 *                            88" . "88
 *                            (| -_- |)
 *                            O\  =  /O
 *                         ____/`---'\____
 *                       .'  \\|     |//  `.
 *                      /  \\|||  :  |||//  \
 *                     /  _||||| -:- |||||-  \
 *                     |   | \\\  -  /// |   |
 *                     | \_|  ''\---/''  |   |
 *                     \  .-\__  `-`  ___/-. /
 *                   ___`. .'  /--.--\  `. . __
 *                ."" '<  `.___\_<|>_/___.'  >'"".
 *               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 *               \  \ `-.   \_ __\ /__ _/   .-` /  /
 *          ======`-.____`-.___\_____/___.-`____.-'======
 *                             `=---='
 *        ^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^~^
 *                     佛祖保佑        永无BUG
**/
#ifndef __DAS_H__
#define __DAS_H__
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


/****************************************************************************************************
 * Micro definition
****************************************************************************************************/
#define TRUE 1
#define FALSE 0
#define INITSIZE 20
#define MAXSIZE 20
#define INCSIZE 10
#define ERROR -9999
#define TESTSIZE 10


/****************************************************************************************************
 * Redefinition
****************************************************************************************************/
typedef int ElemType;
typedef int KeyType;
typedef int DataType;


/****************************************************************************************************
 * Custom function
****************************************************************************************************/
/**
 * @brief	输入一个字符串
 * @param	length [int] 字符串长度
 * @retval	字符串地址
 * @note	实际上的字符串一共有 length + 1 位, 但是最后一个为是结尾, 字符串以 '\0'结尾
**/
ElemType * Enter_String(int length)
{
	int temp = 0;
	ElemType *data = (ElemType *)malloc(sizeof(ElemType) * length + 1);

	printf("Please enter string\n");
	for(int i = 0; i < length; i ++){
		scanf("%d", &temp);
		data[i] = temp;
	}
	data[length] = '\0';

	return data;
}


/**
 * @brief	构造一个字符串
 * @param	data [ElemType **]
 * @note	传入的是二重指针, 因为本函数会改变指针的地址
 * @return	返回值是一个ElemType型的指针, 该指针指向字符串的地址
**/
ElemType * Input_String(void)
{
	ElemType *data;
	int length = 0;

	printf("Please enter the length of your DataStructure\n");
	scanf("%d", &length);

	data = Enter_String(length);

	return data;
}


/**
 * @brief	数组输入函数(不带length版本)
 * @retval	数组地址
**/
ElemType * scanf_string(void)
{
	ElemType temp[100];
	ElemType x = 0;

	int length = 0;

	printf("Please Enter string\n");
	scanf("%d", &x);
	for(int i = 0; i < 100 && x != -9999; i ++){
		temp[i] = x;
		length ++;
		scanf("%d", &x);
	}

	ElemType *data = (ElemType *)malloc(sizeof(ElemType) * (length + 1));

	for(int i = 0; i < length; i ++){
		data[i] = temp[i];
	}
	
	data[length] = '\0';


	return data;
}


#endif