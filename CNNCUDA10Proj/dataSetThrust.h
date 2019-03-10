#ifndef dataSet_H
#define dataSet_H

#include <thrust\host_vector.h>
using namespace std;
using namespace thrust;
class dataSet
{
protected:
	host_vector<host_vector<double>> inputNodes;
	host_vector<host_vector<double>> outputNodes;
	host_vector<int> labels;

public:
	dataSet(size_t numTests);
	dataSet(){}
//	~dataSet();

	size_t getNumSets();
	size_t getInputDimension();
	size_t getOutputDimension();
	host_vector<double>& getInputNodes(size_t choice);
	int getLabel(size_t choice);	//Rename this to getLabel
	host_vector<double>& getOutputNodes(size_t choice);	
	virtual void displayImage(const size_t& choice) {};

};
#endif

