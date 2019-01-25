#include <iostream>
#include <fstream>
#include <sstream>
//#include <opencv2/opencv.hpp>

#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include "imagesAndLabels.h"
#include "CNNStructure.h"
#include "myMathHelpers.h"
#include "simpleDataSet1.h"

using namespace std;

int main() {

//Test vector subtraction overload

/*	string fileNameLabels = "./data/t10k-labels.idx1-ubyte";
	string fileNameImages = "./data/t10k-images.idx3-ubyte";

	imagesAndLabels testImages(fileNameImages, fileNameLabels);

	testImages.displayImage(7);
*/
// Set up a test case for the structure
	vector<int> testCase1;	// Implies two layers of size 3, 2, and 2.
	testCase1.push_back(3);
	testCase1.push_back(3);
	testCase1.push_back(2);

	CNNStructure testStruct(testCase1,.5,1.);	

	cout << "\nShow the matricies " << endl;
	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices(); ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayWeights(iCnt);
	}

	cout << "\nShow the default layerNodes" << endl;
	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices()+1; ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayLayerNodes(iCnt);
	}

// CNN. You start with a set of weights and biases. While you can calculate a cost from that, it does
// not fit into the data you need to calculate the gradient. Calculating the gradient is done over some test
// set. You effectively calculate it for each set but only use the average over the set before you use it to change
// the current weights and biases. The method is called back-propogation because you need a cost at each step,
// and the actual cost is only calculated at the end of the pipeline. As you step backwards a layer at a time, 
// you can produce the needed cost at the next layer. 

// Calculate the gradient. The inputs are the layer nodes, weights, and biases. To get the layer nodes,
// I'll run a data set forward. Then I'll have all the inputs to calculate a gradient, which will be averaged 
// in with the rest. 
// Loop over the training set.

	CNNStructure holdAccumGradients(testCase1);
	vector<double> costHistory;
	simpleDataSet1 data1(testCase1, 40), data2(testCase1, 40);
//Get the starting cost.
	double tempCost = 0.;

	for (size_t tSet = 0; tSet < data1.getSimpleTestSize(); ++tSet) {
		tempCost += testStruct.calcCost(data1.getSimpleTest(tSet), data1.getSimpleNodeLabels(tSet)) / double(data1.getSimpleTestSize());
	}
	cout << "\nStarting cost "<<tempCost;
	costHistory.push_back(tempCost);

//Start training
	size_t numTrainingLoops = 300;
	for (size_t trainLoops = 0; trainLoops < numTrainingLoops; ++trainLoops) {
		for (size_t tSet = 0; tSet < data1.getSimpleTestSize(); ++tSet) {
			testStruct.updateLayers(data1.getSimpleTest(tSet));
			CNNStructure holdTempGradients(testCase1);

// Fill holdTempGradients for this test set. 
			testStruct.makeGradPass(holdTempGradients, data1.getSimpleNodeLabels(tSet));

// Add the temp to the accumulator
			holdAccumGradients += holdTempGradients;
		}
// Divide by the number of entries. You may want to do other things to reduce it as well.
		holdAccumGradients.divideScaler(double(-10.*data1.getSimpleTestSize()));

// Modify the weights and biases.
		testStruct += holdAccumGradients;

// Calculate and store the new cost.To do this, sum up the cost across the entire data set. 
// Note, this not really required for the training. It is so I can get a look at how the
// training progressed. I should see if I could get it cheaply as part of the calculation of 
// the gradient. For example, calculating a cost is not expensive if the layerNodes have
// already been updated. 

		double tempCost = 0.;
		for (size_t tSet = 0; tSet < data1.getSimpleTestSize(); ++tSet) {
			tempCost+=testStruct.calcCost(data1.getSimpleTest(tSet), data1.getSimpleNodeLabels(tSet))/double(data1.getSimpleTestSize());
		}
		costHistory.push_back(tempCost);
	}
	cout << "\nFinal structure " << endl;
	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices(); ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayWeights(iCnt);
	}

	cout << "\nCost history\n";
	cout << costHistory;

// You should now have a trained network. Try it on some cases.
size_t countHits = 0, countMisses = 0;
for (size_t tSet = 0; tSet < data2.getSimpleTestSize(); ++tSet) {
		testStruct.updateLayers(data2.getSimpleTest(tSet));

		cout << "\n simpleTest " << data2.getSimpleTest(tSet);
		cout << " simpleNodes " << data2.getSimpleNodeLabels(tSet);

		testStruct.displayLayerNodes(testStruct.getNumWeightsMatrices());

		if (data2.getSimpleNodeLabels(tSet)[0] == 1) {
			if (testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[0] >
				testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[1]) {
				++countHits;
			}
			else {
				++countMisses;
				cout << "Missed this case 0\n\n";
			}
		}
		else
		{
			if (testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[0] >=
				testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[1]) {
				++countMisses;
				cout << "Missed this case 1\n\n";
			}
			else {
				++countHits;
			}
		}
	}
	cout << "\nNumber of hits == " << countHits << " Number of misses = " << countMisses;
	return 0;
}