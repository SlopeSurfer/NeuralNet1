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

	testImages.displayImage(7);

// Set up a test case for the structure
	vector<int> testCase1;
	testCase1.push_back(5);
	testCase1.push_back(3);
	testCase1.push_back(2);

	CNNStructure testStruct(testCase1);	// Implies three layers of size 5, 3, and 2.

	cout << "\nShow the matricies " << endl;
	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices(); ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayWeights(iCnt);
	}

//Create a gradient structure and add it to the previous. This is how you will update 
//weights and biases.

	CNNStructure testGrad(testCase1);

	cout << "\nShow the gradient matricies " << endl;
	for (size_t iCnt = 0; iCnt < testGrad.getNumWeightsMatrices(); ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testGrad.displayWeights(iCnt);
	}

	testStruct += testGrad;

	cout << "\nShow the revised matricies " << endl;
	for (unsigned int iCnt = 0; iCnt < testStruct.getNumWeightsMatrices(); ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayWeights(iCnt);
	}


	vector<double> input1;
	input1.push_back(.5);
	input1.push_back(.1);
	input1.push_back(.5);
	input1.push_back(.0);
	input1.push_back(.2);

	vector<double> desiredOut1;
	desiredOut1.push_back(.7);
	desiredOut1.push_back(.2);


cout << "\nCalc " << testStruct.calcCost(input1, desiredOut1)<<"\n";

	return 0;
}