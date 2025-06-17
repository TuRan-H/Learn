// @before-stub-for-debug-begin
#include "commoncppproblem27.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=27 lang=cpp
 *
 * [27] 移除元素
 */

// @lc code=start
class Solution {
public:
	int removeElement(vector<int> &nums, int val)
	{
		int left = 0, right = nums.size() - 1;

		// ! 边界条件, nums为空
		if (!nums.size())
			return 0;

		while (left < right) {
			if (nums[left] == val) {
				nums[left] = nums[right];
				--right;
			} else {
				++left;
			}
		}

		if (nums[left] == val)
			return left;
		else
			return left + 1;
	}
};
// @lc code=end
