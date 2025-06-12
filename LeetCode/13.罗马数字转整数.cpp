// @before-stub-for-debug-begin
#include "commoncppproblem13.h"
#include <string>
#include <vector>
#include <unordered_map>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=13 lang=cpp
 *
 * [13] 罗马数字转整数
 */

// @lc code=start
class Solution {
public:
	int romanToInt(string s)
	{
		// Using std::string for clarity
		std::unordered_map<char, int> roman_map = {
			{ 'I', 1 },
			{ 'V', 5 },
			{ 'X', 10 },
			{ 'L', 50 },
			{ 'C', 100 },
			{ 'D', 500 },
			{ 'M', 1000 }
		};

		int total = 0;
		int n = s.length();
		for (int i = 0; i < n; ++i) {
			// Check for subtractive notation
			// If the current symbol's value is less than the next symbol's value
			if (i + 1 < n && roman_map[s[i]] < roman_map[s[i + 1]]) {
				total += roman_map[s[i + 1]] - roman_map[s[i]];
				i++; // Increment i an extra time because we've processed two characters
			} else {
				// Additive notation
				total += roman_map[s[i]];
			}
		}
		return total;
	}
};
// @lc code=end