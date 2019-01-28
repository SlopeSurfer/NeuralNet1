#ifndef dataSet_H
#define dataSet_H

#include <vector>
using namespace std;
class dataSet
{
protected:
	vector<vector<double>> inputNodes;
	vector<vector<double>> outputNodes;
	vector<int> labels;

public:
	dataSet(size_t numTests);
	dataSet(){}
//	~dataSet();

	size_t getNumSets();
	size_t getInputDimension();
	size_t getOutputDimension();
	vector<double> getInputNodes(size_t choice);
	int getLabels(size_t choice);
	vector<double> getOutputNodes(size_t choice);
	virtual void displayImage(const size_t& index) {};

};
#endif

