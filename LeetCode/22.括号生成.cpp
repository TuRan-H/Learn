#include <string>
#include <vector>
using namespace std;
/*
 * @lc app=leetcode.cn id=22 lang=cpp
 *
 * [22] 括号生成
 */

// @lc code=start
class Solution {
public:
	vector<string> generateParenthesis(int n)
	{
		vector<string> result;
		string current;

		backtrack(result, current, 0, 0, n);

		return result;
	}

private:
	// 回溯算法
	void backtrack(vector<string>& result, string& current, int left, int right, int n)
	{
		if (left + right == n * 2) {
			result.push_back(current);
			return;
		}

		if (left < n) {
			current.push_back('(');
			backtrack(result, current, left + 1, right, n);
			current.pop_back();
		}

		if (left > right) {
			current.push_back(')');
			backtrack(result, current, left, right + 1, n);
			current.pop_back();
		}
	}
};
// @lc code=end
