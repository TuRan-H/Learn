// @before-stub-for-debug-begin
#include "commoncppproblem18.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=18 lang=cpp
 *
 * [18] 四数之和
 */

// 解题方案: 固定前两个数, 通过两数之和的双指针方法求解

// @lc code=start
class Solution {
public:
	vector<vector<int>> fourSum(vector<int>& nums, int target)
	{
		vector<vector<int>> quadruples;

		if (nums.size() < 4)
			return quadruples;

		// 对数组进行排序
		sort(nums.begin(), nums.end());

		for (int i = 0; i < nums.size(); ++i) {
			// ! 防止nums[i] 和 nums[i - 1] 完全相同, 这样可能会产生重复的三元组
			if (i > 0 && nums[i] == nums[i - 1])
				continue;

			for (int j = i + 1; j < nums.size(); ++j) {
				// 同理, 防止nums[j] 和 nums[j - 1] 完全相同
				if (j > i + 1 && nums[j] == nums[j - 1])
					continue;

				// ! 这需要将new_target转换成long类型, 因为new_target可能会超出int类型的数值上下限
				long new_target = (long)target - (long)nums[i] - (long)nums[j];

				int p_left = j + 1, p_right = nums.size() - 1;
				while (p_left < p_right) {
					if ((long)nums[p_left] + nums[p_right] < new_target)
						++p_left;
					else if ((long)nums[p_left] + nums[p_right] > new_target)
						--p_right;
					else {
						quadruples.push_back({ nums[i], nums[j], nums[p_left], nums[p_right] });

						// ! p_left向右移动, 跳过重复的元素, p_right向左移动, 跳过重复的元素
						while (p_left < p_right && nums[p_left + 1] == nums[p_left])
							++p_left;
						while (p_left < p_right && nums[p_right - 1] == nums[p_right])
							--p_right;

						// ! 此时, p_left的下一个元素以及p_right的前一个元素与p_left和p_right不一样
						// ! 指针前进一步
						++p_left;
						--p_right;
					}
				}
			}
		}

		return quadruples;
	}
};
// @lc code=end
