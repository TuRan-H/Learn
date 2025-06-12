#include <limits>
#include <vector>

using namespace std;

/*
 * @lc app=leetcode.cn id=4 lang=cpp
 *
 * [4] 寻找两个正序数组的中位数
 */

// @lc code=start
class Solution {
public:
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
	{
		int m = nums1.size();
		int n = nums2.size();

		if (m > n) {
			std::swap(nums1, nums2);
			std::swap(m, n);
		}

		int low = 0;
		int high = m;

		int totalLeftElements = (m + n + 1) / 2;

		double median = 0.0;

		while (low <= high) {
			int partitionX = low + (high - low) / 2;

			int partitionY = totalLeftElements - partitionX;

			int maxLeftX = (partitionX == 0) ? std::numeric_limits<int>::min() : nums1[partitionX - 1];
			int minRightX = (partitionX == m) ? std::numeric_limits<int>::max() : nums1[partitionX];

			int maxLeftY = (partitionY == 0) ? std::numeric_limits<int>::min() : nums2[partitionY - 1];
			int minRightY = (partitionY == n) ? std::numeric_limits<int>::max() : nums2[partitionY];

			if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
				if ((m + n) % 2 == 0) {
					median = (std::max(maxLeftX, maxLeftY) + std::min(minRightX, minRightY)) / 2.0;
				} else {
					median = static_cast<double>(std::max(maxLeftX, maxLeftY));
				}
				break;
			} else if (maxLeftX > minRightY) {
				high = partitionX - 1;
			} else {
				low = partitionX + 1;
			}
		}
		return median;
	}
};
// @lc code=end
