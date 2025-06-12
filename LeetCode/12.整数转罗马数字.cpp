// @before-stub-for-debug-begin
#include "commoncppproblem12.h"
#include <map>
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=12 lang=cpp
 *
 * [12] 整数转罗马数字
 */

// @lc code=start
class Solution {
public:
	string intToRoman(int num)
	{
		// 按位拆分
		vector<int> decomposing_result = this->decomposing(num);

		// 数值 -> 符号的转换
		// 将按位拆分的结果作为数组转入, 输出字符串
		string result;
		for (auto& i : decomposing_result) {
			result += this->i2r(i);
		}

		return result;
	}

	vector<int> decomposing(int num)
	{
		vector<int> result;
		int multiplying = 1;

		while (num > 10) {
			int remainder = num % 10;
			result.insert(result.begin(), remainder * multiplying);
			multiplying *= 10;
			num = (num - remainder) / 10;
		}

		result.insert(result.begin(), num * multiplying);

		if (!result.empty() && *result.rbegin() == 0)
			result.pop_back();

		return result;
	}

	string i2r(int num)
	{
		if (i2r_map.find(num) != i2r_map.end()) {
			return i2r_map[num];
		}

		// 如果不在映射表中, 则使用贪心算法求解
		string result = "";

		vector<pair<int, string>> romanPairs = {
			{ 1000, "M" }, { 900, "CM" }, { 500, "D" }, { 400, "CD" },
			{ 100, "C" }, { 90, "XC" }, { 50, "L" }, { 40, "XL" },
			{ 10, "X" }, { 9, "IX" }, { 5, "V" }, { 4, "IV" }, { 1, "I" }
		};

		for (auto& pair : romanPairs) {
			while (num >= pair.first) {
				result += pair.second;
				num -= pair.first;
			}
		}

		return result;
	}

private:
	map<int, string> i2r_map = {
		{ 1000, "M" }, { 900, "CM" }, { 500, "D" }, { 400, "CD" },
		{ 100, "C" }, { 90, "XC" }, { 50, "L" }, { 40, "XL" },
		{ 10, "X" }, { 9, "IX" }, { 5, "V" }, { 4, "IV" }, { 1, "I" }
	};
};
// @lc code=end
