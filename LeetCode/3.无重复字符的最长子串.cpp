#include <string>
#include <vector>

using namespace std;

/*
 * @lc app=leetcode.cn id=3 lang=cpp
 *
 * [3] 无重复字符的最长子串
 */

// @lc code=start
class Solution {
public:
	int lengthOfLongestSubstring(string s)
	{
		int n = s.length();
		if (n == 0) {
			return 0;
		}

		// 记录每个字符最后出现的位置，初始化为-1
		std::vector<int> lastIndex(256, -1);

		int maxLength = 0; // 最长子串长度
		int left = 0; // 滑动窗口左边界

		for (int right = 0; right < n; ++right) {
			char currentChar = s[right];

			// 如果当前字符在窗口内重复出现，移动左边界
			if (lastIndex[static_cast<unsigned char>(currentChar)] != -1 && lastIndex[static_cast<unsigned char>(currentChar)] >= left) {
				left = lastIndex[static_cast<unsigned char>(currentChar)] + 1;
			}

			// 更新字符最后出现位置
			lastIndex[static_cast<unsigned char>(currentChar)] = right;

			// 更新最大长度
			maxLength = std::max(maxLength, right - left + 1);
		}

		return maxLength;
	}
};
// @lc code=end
