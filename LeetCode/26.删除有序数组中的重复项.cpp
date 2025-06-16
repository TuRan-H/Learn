// @before-stub-for-debug-begin
#include "commoncppproblem26.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=26 lang=cpp
 *
 * [26] 删除有序数组中的重复项
 */

// @lc code=start
class Solution {
public:
	int removeDuplicates(vector<int> &nums)
	{
		int i = 0;

		while (i < nums.size()) {
			if (i < nums.size() - 1 && nums[i] > nums[i + 1]) {
				break;
			}

			if (i > 0 && nums[i] == nums[i - 1]) {
				int temp = nums[i];
				nums.erase(nums.begin() + i);
			} else {
				i++;
			}
		}


		return nums.size();
	}
};
// @lc code=end
