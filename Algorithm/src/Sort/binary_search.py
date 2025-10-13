"""
二分查找 binary search
"""


def find_first_num_index(input_list, target, left, right):
    if left > right:
        return -1

    mid = int((left + right) / 2)

    if input_list[mid] == target:
        if mid == 0 or input_list[mid - 1] != target:
            return mid
        else:
            return find_first_num_index(input_list, target, left, mid - 1)

    if input_list[mid] < target:
        return find_first_num_index(input_list, target, mid + 1, right)
    else:
        return find_first_num_index(input_list, target, left, mid - 1)


if __name__ == "__main__":
    input_list = [1, 2, 3, 4, 5]
    target = 10
    result = find_first_num_index(input_list, target, 0, len(input_list) - 1)
    print(result)
