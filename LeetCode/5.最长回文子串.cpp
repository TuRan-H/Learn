#include <string>

using namespace std;

/*
 * @lc app=leetcode.cn id=5 lang=cpp
 *
 * [5] 最长回文子串
 */

// @lc code=start
class Solution {
public:
	void expandAndCheck(const std::string& s, int left_candidate, int right_candidate,
		int& current_max_start, int& current_max_len)
	{
		int l = left_candidate;
		int r = right_candidate;

		while (l >= 0 && r < s.length() && s[l] == s[r]) {
			l--;
			r++;
		}

		int current_palindrome_len = r - l - 1;

		if (current_palindrome_len > current_max_len) {
			current_max_len = current_palindrome_len;
			current_max_start = l + 1;
		}
	}

	std::string longestPalindrome(std::string s)
	{
		if (s.length() < 1) {
			return "";
		}

		int res_start_idx = 0;
		int res_max_len = 1;

		for (int i = 0; i < s.length(); ++i) {
			expandAndCheck(s, i, i, res_start_idx, res_max_len);

			expandAndCheck(s, i, i + 1, res_start_idx, res_max_len);
		}

		return s.substr(res_start_idx, res_max_len);
	}
};
// @lc code=end
