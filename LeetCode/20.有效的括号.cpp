// @before-stub-for-debug-begin
#include "commoncppproblem20.h"
#include <set>
#include <stack>
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=20 lang=cpp
 *
 * [20] 有效的括号
 */

// @lc code=start
class Solution {
public:
	bool isValid(string s)
	{
		stack<char> brackets;

		for (auto& c : s) {
			if (c == '(' or c == '[' or c == '{') {
				brackets.push(c);
			}

			if (c == ')' or c == ']' or c == '}') {
				if (brackets.empty())
					return false;
				else {
					char top = brackets.top();
					brackets.pop();
					if (c == ')' && top != '(' || c == ']' && top != '[' || c == '}' && top != '{')
						return false;
				}
			}
		}

		if (brackets.empty())
			return true;
		else
			return false;
	}
};
// @lc code=end
