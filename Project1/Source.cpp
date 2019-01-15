#include <iostream>
#include <fstream>
#include <sstream>
//#include <opencv2/opencv.hpp>

#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include "imagesAndLabels.h"
#include "CNNStructure.h"
#include "myMathHelpers.h"

using namespace std;

int main() {

	string fileNameLabels = "./data/t10k-labels.idx1-ubyte";
	string fileNameImages = "./data/t10k-images.idx3-ubyte";

	imagesAndLabels testImages(fileNameImages, fileNameLabels);

	testImages.displayImage(5);

// Set up a test case for the structure
	vector<int> testCase1;
	testCase1.push_back(3);
	testCase1.push_back(2);
	testCase1.push_back(2);

	CNNStructure testStruct(testCase1);

	vector<double> input1;
	input1.push_back(.5);
	input1.push_back(.1);
	input1.push_back(.5);

	vector<double> desiredOut1;
	desiredOut1.push_back(.7);
	desiredOut1.push_back(.2);

cout << "\nCalc 1 " << testStruct.calcCost(input1, desiredOut1);

input1.push_back(1.);	// To pick up the bias terms
vector<double> vecBack = vecMatMult(input1, testStruct.getSingleMat());
cout << "\nThe two a values " << vecBack[0] << " " << vecBack[1];
double cost = 0, tempSum = 0;
for (int iCnt = 0; iCnt < 2; iCnt++) {
	tempSum += pow((desiredOut1[iCnt] - vecBack[iCnt]),2);
}
cout << "\nCost by singleMat method "<<tempSum;

	return 0;
}