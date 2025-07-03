// @before-stub-for-debug-begin
#include "commoncppproblem29.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=29 lang=cpp
 *
 * [29] 两数相除
 */

// @lc code=start
class Solution {
public:
	int divide(int dividend, int divisor)
	{
		int sign = ((dividend ^ divisor) >> 31 & 0x1) ? -1 : 1;
		long result = 0;
		long dividend_long = abs((long)dividend);
		long divisor_long = abs((long)divisor);
		while (dividend_long >= divisor_long) {
			long i = 1;
			long tmp = divisor_long;
			while (dividend_long >= tmp) {
				dividend_long -= tmp;
				result += i;
				tmp = tmp << 1;
				i = i << 1;
			}
		}

		result *= sign;
		if (result > INT32_MAX) {
			return INT32_MAX;
		}
		if (result < INT32_MIN) {
			return INT32_MIN;
		}

		return (int)result;
	}
};
// @lc code=end