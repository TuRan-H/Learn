// @before-stub-for-debug-begin
#include "commoncppproblem15.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=15 lang=cpp
 *
 * [15] 三数之和
 */

/* 
 * 解法: 固定三元组中某一个数, 将剩下来两个数转化为 两数之和 问题
 */

// @lc code=start
class Solution {
public:
	vector<vector<int>> threeSum(vector<int>& nums)
	{
		vector<vector<int>> triples = {};
		sort(nums.begin(), nums.end());

		for (int i = 0; i < nums.size(); ++i) {
			if (i > 0 && nums[i] == nums[i - 1])
				continue;
			int target = -nums[i];
			int k = nums.size() - 1;

			for (int j = i + 1; j < nums.size(); ++j) {
				if (j > i + 1 && nums[j] == nums[j -1])
					continue;

				while (j < k && nums[j] + nums[k] > target)
					--k;

				if (j == k)
					break;

				if (nums[j] + nums[k] == target)
					triples.push_back({ nums[i], nums[j], nums[k] });
			}
		}
		
		return triples;
	}
};
// @lc code=end
