#include "dataSetThrust.h"
#include <assert.h>
using namespace thrust;
dataSet::dataSet(size_t numTests)
{
//I'd like a test example that is small enough that I can calculate some 
// of it by hand. I'm going to feed it host_vectors that have random numbers between 0 and 1. 
// if a number is less than .5 it will be considered a 0. Numbers .5 and greater will be considered 1.
// Then I will use the ones and zeros as a 3 digit binary representation.  Numbers 0,1,2, and 3 will 
// be labled as 0. And numbers 4,5,6, and 7 will be labled as 1. If it is 0, I want y0 = 1 and y1 = 0.
// If it is 1, y0 = 0 and y1 = 1;
// Note also that the original node layer should have the extra 1. at the end. 
	double nextNum;
	int tempLabel, tempBit;
	for (int testCnt = 0; testCnt < numTests; ++testCnt) {
		host_vector<double> eachVec;
		tempLabel = 0;
		for (int iCnt = 0; iCnt <3; iCnt++) {
			nextNum = float(rand() % 10) / 10.;
			eachVec.push_back(nextNum);
			tempBit = int((nextNum + .5));
			tempLabel += int(tempBit*pow(2, 2 - iCnt));
		}
		host_vector<double>tempVecLabel;
		if (tempLabel < 4) {
			tempLabel = 0;
			tempVecLabel.push_back(1);
			tempVecLabel.push_back(0);
			tempVecLabel.push_back(1);
		}
		else
		{
			tempLabel = 1;
			tempVecLabel.push_back(0);
			tempVecLabel.push_back(1);
			tempVecLabel.push_back(1);
		}
		eachVec.push_back(1.);
		inputNodes.push_back(eachVec);
		labels.push_back(tempLabel);
		outputNodes.push_back(tempVecLabel);
	}
}
/*
dataSet::~dataSet()
{
}
*/

size_t dataSet::getNumSets() {
	return inputNodes.size();
}
host_vector<double>& dataSet::getInputNodes(size_t choice) {
	assert(choice >= 0 && choice < inputNodes.size());
	return inputNodes[choice];
}

int dataSet::getLabel(size_t choice) {
	assert(choice >= 0 && choice < labels.size());
	return labels[choice];
}

host_vector<double>& dataSet::getOutputNodes(size_t choice) {
	assert(choice >= 0 && choice < outputNodes.size());
	return outputNodes[choice];
}

size_t dataSet::getInputDimension(){
	return inputNodes[0].size();
}

size_t dataSet::getOutputDimension() {
	return outputNodes[0].size();

}