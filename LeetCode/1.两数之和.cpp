#include <unordered_map>
#include <vector>

using namespace std;

/*
 * @lc app=leetcode.cn id=1 lang=cpp
 *
 * [1] 两数之和
 */

// @lc code=start
class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target)
	{
		// 存储数字和其索引
		std::unordered_map<int, int> num_map;

		for (int i = 0; i < nums.size(); ++i) {
			int current_num = nums[i];
			int complement = target - current_num;

			// 检查补数是否存在于哈希表中
			if (num_map.count(complement)) {
				// 如果找到，返回补数和当前数字的索引
				return { num_map[complement], i };
			}

			// 如果没有找到补数，将当前数字和其索引添加到哈希表中
			num_map[current_num] = i;
		}

		return {};
	}
};
// @lc code=end
