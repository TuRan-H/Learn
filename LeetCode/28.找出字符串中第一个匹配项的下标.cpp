// @before-stub-for-debug-begin
#include "commoncppproblem28.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=28 lang=cpp
 *
 * [28] 找出字符串中第一个匹配项的下标
 */

// @lc code=start
class Solution {
public:
	int strStr(string haystack, string needle)
	{
		int p_h = 0;
		bool success = true;

		while (p_h < haystack.size()) {
			for (int i = 0; i < needle.size(); ++i) {
				if (haystack[p_h + i] == needle[i])
					success = true;
				else {
					success = false;
					break;
				}
			}

			if (success == false)
				++p_h;
			else
				return p_h;
		}

		return -1;
	}
};
// @lc code=end
