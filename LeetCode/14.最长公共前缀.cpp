#include <string>
#include <vector>
using namespace std;

/*
 * @lc app=leetcode.cn id=14 lang=cpp
 *
 * [14] 最长公共前缀
 */

// @lc code=start
class Solution {
public:
	string longestCommonPrefix(vector<string>& strs)
	{
		int len = strs.size();

		// 判断出最短的那一个子串长度
		int min_size = 200;
		for (auto& s : strs) {
			if (s.size() < min_size)
				min_size = s.size();
		}

		// 迭代子串中的第 i 个元素
		for (int i = 0; i < min_size; ++i) {
			char current_char = strs[0][i];
			// 迭代每一个子串 s
			for (auto& s : strs) {
				if (s[i] != current_char)
					return s.substr(0, i);
			}
		}

		return strs[0].substr(0, min_size);
	};
};
// @lc code=end
