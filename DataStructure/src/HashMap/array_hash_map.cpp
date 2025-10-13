// 基于数组的hash表实现

#include <iostream>
#include <string>
#include <vector>
using namespace std;

// 实现一个键值对数据结构
class Pair {
public:
	int key;
	string value;

	Pair(int k, string v)
		: key(k)
		, value(v)
	{
		this->key = k;
		this->value = v;
	}
};

// 实现一个基于数组的哈希表
class ArrayHashMap {
private:
	vector<Pair *> buckets;

public:
	ArrayHashMap()
	{
		buckets = vector<Pair *>(100);
	};

	~ArrayHashMap()
	{
		// pair是一个常量指针的引用, 不可以修改指针所指向的地址, 但是可以修改指针所指向的值
		for (const auto &pair : buckets) {
			delete pair;
		}
	}

	int hashFunc(int key)
	{
		int index = key % 100;
		return index;
	}

	/**
	 * @brief 获取一个键值对
	 */
	string get(int key)
	{
		int index = hashFunc(key);
		Pair *pair = buckets[index];
		if (pair == nullptr)
			return "";

		return pair->value;
	}

	/**
	 * @brief 插入一个键值对
	 */
	void put(int key, string value)
	{
		int index = hashFunc(key);
		if (buckets[index] == nullptr) {
			buckets[index] = new Pair(key, value);
		} else {
			buckets[index]->value = value;
		}
	}

	/**
	 * @brief 删除一个键值对
	 */
	void remove(int key)
	{
		int index = hashFunc(key);
		delete buckets[index];
		buckets[index] = nullptr;
	}

	/**
	 * @brief 返回所有键值对
	 */
	vector<Pair *> pairSet()
	{
		vector<Pair *> pairSet;

		for (Pair *const &pair : buckets) {
			if (pair != nullptr) {
				pairSet.push_back(pair);
			}
		}

		return pairSet;
	}

	void print()
	{
		for (const auto &pair : pairSet()) {
			cout << pair->key << ": " << pair->value << endl;
		}
	}
};

int main()
{
	ArrayHashMap hashMap;

    hashMap.put(1, "Apple");
    hashMap.put(2, "Banana");
    hashMap.put(15, "Cherry");  // 15 % 100 = 15
    hashMap.put(101, "Date");   // 101 % 100 = 1 (会覆盖Apple)

	hashMap.print();

	return 0;
}