// @before-stub-for-debug-begin
#include "commoncppproblem17.h"
#include <map>
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=17 lang=cpp
 *
 * [17] 电话号码的字母组合
 */

// @lc code=start
class Solution {
public:
	vector<string> letterCombinations(string digits)
	{
		vector<string> letter_combine;

		if (digits.size() == 0) {
			return vector<string>();
		}

		int result_length = 1;

		for (auto d_it = digits.rbegin(); d_it != digits.rend(); ++d_it) {
			// result_length *= this->digitsMap[d].size();
			if (letter_combine.size() == 0) {
				for (auto& a : this->digitsMap[*d_it])
					letter_combine.push_back(string(1, a));
				continue;
			}

			vector<string> new_letter_combine;
			for (auto& a : this->digitsMap[*d_it]) {
				for (auto& lc : letter_combine) {
					new_letter_combine.push_back(a + lc);
				}
			}

			letter_combine = new_letter_combine;
		}

		return letter_combine;
	}

private:
	map<char, string> digitsMap = {
		{ '2', "abc" },
		{ '3', "def" },
		{ '4', "ghi" },
		{ '5', "jkl" },
		{ '6', "mno" },
		{ '7', "pqrs" },
		{ '8', "tuv" },
		{ '9', "wxyz" }
	};
};
// @lc code=end
