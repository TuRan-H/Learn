"""
最长公共子序列算法 logest common subsequence algorithm
"""
import numpy as np

def longest_common_subsequence(s1: str, s2: str) -> str:
    """
    计算并返回两个字符串的最长公共子序列
    """
    D = np.zeros((len(s1)+1, len(s2)+1), dtype=int)

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                D[i][j] = 1 + D[i-1][j-1]
            else:
                D[i][j] = max(D[i-1][j], D[i][j-1])

    lcs = []
    i = len(s1)
    j = len(s2)
    while i>0 and j>0:
        if s1[i-1] == s2[j-1]:
            i -= 1
            j -= 1
            lcs.append(s1[i])
        else:
            if D[i-1][j] >= D[i][j-1]:
                i -= 1
            else:
                j -= 1

    lcs.reverse()
    return ''.join(lcs)

if __name__ == "__main__":
	# 使用函数
	s1 = 'ABCBDAB'
	s2 = 'BDCABC'
	print(longest_common_subsequence(s1, s2))