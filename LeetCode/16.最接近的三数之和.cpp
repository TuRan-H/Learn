// @before-stub-for-debug-begin
#include "commoncppproblem16.h"
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=16 lang=cpp
 *
 * [16] 最接近的三数之和
 */

// @lc code=start
class Solution {
public:
	int threeSumClosest(vector<int>& nums, int target)
	{
		sort(nums.begin(), nums.end());

		// 返回值, 三数之和
		int closest_sum = 0;
		// min代表三数之和与target的差的绝对值 $ |n1 + n2 + n3 - target| $
		int min_diff = numeric_limits<int>::max();
		for (int i = 0; i < nums.size(); ++i) {
			int new_target = target - nums[i];

			int k = nums.size() - 1;
			int j = i + 1;
			while (j < k) {
				int current_diff = abs(nums[k] + nums[j] - new_target);
				if (current_diff < min_diff) {
					min_diff = current_diff;
					closest_sum = nums[i] + nums[j] + nums[k];
				}

				if (nums[i] + nums[j] + nums[k] < target)
					++j;
				else if (nums[i] + nums[j] + nums[k] > target)
					--k;
				else
					return closest_sum;
			}
		}

		return closest_sum;
	}
};
// @lc code=end
