/**
 * 基于链式寻址的hash表
 * 能够有效解决hash冲突的问题
 */

#include <iostream>
#include <vector>
using namespace std;

/**
 * KV pair
 */
struct Pair {
public:
	int key;
	string val;
	Pair(int key, string val)
	{
		this->key = key;
		this->val = val;
	}
};

/**
 * @brief 链式地址 hash 表
 */
class HashMapChaining {
private:
	int size; // 键值对数量
	int capacity; // 哈希表容量
	double loadThres; // 触发扩容的负载因子阈值
	int extendRatio; // 扩容倍数
	vector<vector<Pair *>> buckets; // 桶数组

public:
	HashMapChaining()
		: size(0)
		, capacity(4)
		, loadThres(2.0 / 3.0)
		, extendRatio(2)
	{
		buckets.resize(capacity);
	}

	~HashMapChaining()
	{
		for (auto &bucket : buckets) {
			for (auto *pair : bucket) {
				delete pair;
			}
		}
	}

	/**
	 * @brief 哈希函数
	 */
	int hashFunc(int key)
	{
		return key % capacity;
	}

	/**
	 * @brief 计算当前负载因子
	 */
	double loadFactor()
	{
		return (double)size / (double)capacity;
	}

	/**
	 * @brief 查找操作
	 */
	string get(int key)
	{
		int index = hashFunc(key);
		// 遍历桶，若找到 key ，则返回对应 val
		for (Pair *pair : buckets[index]) {
			if (pair->key == key) {
				return pair->val;
			}
		}
		// 若未找到 key ，则返回空字符串
		return "";
	}

	/**
	 * @brief 添加操作
	 */
	void put(int key, string val)
	{
		// 当负载因子超过阈值时，执行扩容
		if (loadFactor() > loadThres) {
			extend();
		}
		int index = hashFunc(key);
		// 遍历桶，若遇到指定 key ，则更新对应 val 并返回
		for (Pair *pair : buckets[index]) {
			if (pair->key == key) {
				pair->val = val;
				return;
			}
		}
		// 若无该 key ，则将键值对添加至尾部
		buckets[index].push_back(new Pair(key, val));
		size++;
	}

	/**
	 * @brief 删除操作
	 */
	void remove(int key)
	{
		int index = hashFunc(key);
		auto &bucket = buckets[index];
		// 遍历桶，从中删除键值对
		for (int i = 0; i < bucket.size(); i++) {
			if (bucket[i]->key == key) {
				Pair *tmp = bucket[i];
				bucket.erase(bucket.begin() + i); // 从中删除键值对
				delete tmp; // 释放内存
				size--;
				return;
			}
		}
	}

	/**
	 * @brief 扩容哈希表
	 */
	void extend()
	{
		// 暂存原哈希表
		vector<vector<Pair *>> bucketsTmp = buckets;
		// 初始化扩容后的新哈希表
		capacity *= extendRatio;
		buckets.clear();
		buckets.resize(capacity);
		size = 0;
		// 将键值对从原哈希表搬运至新哈希表
		for (auto &bucket : bucketsTmp) {
			for (Pair *pair : bucket) {
				put(pair->key, pair->val);
				// 释放内存
				delete pair;
			}
		}
	}

	/**
	 * @brief 打印哈希表
	 */
	void print()
	{
		for (auto &bucket : buckets) {
			cout << "[";
			for (Pair *pair : bucket) {
				cout << pair->key << " -> " << pair->val << ", ";
			}
			cout << "]\n";
		}
	}
};

int main()
{
	HashMapChaining hash_map;
	hash_map.put(1, "one");
	hash_map.put(5, "two");
	hash_map.put(9, "three");
	hash_map.put(405, "four");
	// hash_map.print();
	string tmp = hash_map.get(405);
	printf("key: 405, val: %s\n", tmp.c_str());

	hash_map.print();
	printf("hello world");
	printf("hello world");
}