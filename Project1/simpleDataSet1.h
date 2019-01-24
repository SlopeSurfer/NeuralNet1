#ifndef simpleDataSet1_H
#define simpleDataSet1_H

#include <vector>
using namespace std;
class simpleDataSet1
{
	vector<vector<double>> simpleTest;
	vector<int> simpleLabels;
	vector<vector<double>> simpleNodeLabels;
public:
	simpleDataSet1(vector<int> testStructDescription, size_t numTests);
//	~simpleDataSet1();

	size_t getSimpleTestSize();
	vector<double> getSimpleTest(size_t choice);
	int getSimpleLabels(size_t choice);
	vector<double> getSimpleNodeLabels(size_t choice);

};
#endif

