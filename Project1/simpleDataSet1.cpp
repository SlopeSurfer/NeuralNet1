#include "simpleDataSet1.h"
#include <assert.h>
simpleDataSet1::simpleDataSet1(vector<int> testStructDescription,size_t numTests)
{
//I'd like a test example that is small enough that I can calculate some 
// of it by hand. I'm going to feed it vectors that have random numbers between 0 and 1. 
// if a number is less than .5 it will be considered a 0. Numbers .5 and greater will be considered 1.
// Then I will use the ones and zeros as a 3 digit binary representation.  Numbers 0,1,2, and 3 will 
// be labled as 0. And numbers 4,5,6, and 7 will be labled as 1. If it is 0, I want y0 = 1 and y1 = 0.
// If it is 1, y0 = 0 and y1 = 1;
// Note also that the original node layer should have the extra 1. at the end. 
	double nextNum;
	int tempLabel, tempBit;

	for (int testCnt = 0; testCnt < numTests; ++testCnt) {
		vector<double> eachVec;
		tempLabel = 0;
		for (int iCnt = 0; iCnt < testStructDescription[0]; iCnt++) {
			nextNum = float(rand() % 10) / 10.;
			eachVec.push_back(nextNum);
			tempBit = int((nextNum + .5));
			tempLabel += int(tempBit*pow(2, 2 - iCnt));
		}
		vector<double>tempVecLabel;
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
		simpleTest.push_back(eachVec);
		simpleLabels.push_back(tempLabel);
		simpleNodeLabels.push_back(tempVecLabel);
	}

}

/*
simpleDataSet1::~simpleDataSet1()
{
}
*/

size_t simpleDataSet1::getSimpleTestSize() {
	return simpleTest.size();
}
vector<double> simpleDataSet1::getSimpleTest(size_t choice) {
	assert(choice >= 0 && choice < simpleTest.size());
	return simpleTest[choice];
}

int simpleDataSet1::getSimpleLabels(size_t choice) {
	assert(choice >= 0 && choice < simpleLabels.size());
	return simpleLabels[choice];
}

vector<double> simpleDataSet1::getSimpleNodeLabels(size_t choice) {
	assert(choice >= 0 && choice < simpleNodeLabels.size());
	return simpleNodeLabels[choice];
}
