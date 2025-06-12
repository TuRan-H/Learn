// @before-stub-for-debug-begin
#include "commoncppproblem11.h"
#include <string>
#include <vector>

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=11 lang=cpp
 *
 * [11] 盛最多水的容器
 */

// @lc code=start
class Solution {
public:
	int maxArea(vector<int>& height)
	{
		int max_area = 0;
		int area = 0;

		int i = 0, j = height.size() - 1;
		while (i < j) {
			area = (j - i) * std::min(height.at(i), height.at(j));
			max_area = std::max(area, max_area);
			if (height[i] < height[j])
				++i;
			else
				--j;
		}

		return max_area;
	};
};
// @lc code=end
