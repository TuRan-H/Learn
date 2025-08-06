#include <vector>
using namespace std;

class HashMapChaining {
private:
	int size; // 键值对数量
	int capacity; // 哈希表容量 (桶的大小)
	double loadThres; // 负载因子的阈值
	int extendRatio; // 扩容倍数
	vector<vector<Pair *>> buckets;
}