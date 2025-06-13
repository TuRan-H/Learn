#include <iostream>

int recur(int n, int res)
{
	if (n == 0)
		return res;

	return recur(n - 1, res + n);
}

int main()
{
	std::cout << recur(10, 10) << std::endl;

	return 0;
}