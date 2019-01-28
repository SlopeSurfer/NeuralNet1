#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include "imagesAndLabels.h"
#include "CNNStructure.h"
#include "myMathHelpers.h"
#include "dataSet.h"
#include "handNumberData.h"
using namespace std;

int main() {

//Set up the training data.
	size_t numToSave;
	double gradientCutDown = 50.;
	size_t lapCounter = 0,numBetweenPrints = 9,numSinceLastPrint = 0;
	// Set up a test case for the structure
	vector<int> testCase;

	string fileNameLabels = "./data/t10k-labels.idx1-ubyte";
	string fileNameImages = "./data/t10k-images.idx3-ubyte";

	handNumberData data1(fileNameImages, fileNameLabels);
	data1.displayImage(5);
	testCase.push_back((int)data1.getInputDimension());
	testCase.push_back(16);
	testCase.push_back(16);
	testCase.push_back((int)data1.getOutputDimension() );

	numToSave = 200;
/*
	dataSet data1(80), data2(40);
	
	testCase.push_back((int)data1.getInputDimension());
	testCase.push_back(3);
	testCase.push_back(3);
	testCase.push_back((int)data1.getOutputDimension());
	numToSave = 40;
*/	
	cout << "\ndata1.getOutputDimension() " << data1.getOutputDimension();
	cout << "\ndata1.getInputDimension() " << data1.getInputDimension();
	
//	CNNStructure testStruct(testCase, .5, 1.);
	CNNStructure testStruct("./states/weightsFile4.txt");

//	testStruct.displayWeights(1);
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

#ifdef _DEBUG

	cout << "\nShow the matricies " << endl;
	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices(); ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayWeights(iCnt);
	}

	cout << "\nShow the default layerNodes" << endl;
	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices() + 1; ++iCnt) {
		cout << "\n Layer " << iCnt << endl;
		testStruct.displayLayerNodes(iCnt);
	}
#endif // !_DEBUG
	CNNStructure holdAccumGradients(testCase);
	vector<double> costHistory;
//Get the starting cost.
	double tempCost = 0.;

	for (size_t tSet = 0; tSet < data1.getNumSets()-numToSave; ++tSet) {
		tempCost += testStruct.calcCost(data1.getInputNodes(tSet), data1.getOutputNodes(tSet)) / double(data1.getNumSets()-numToSave); 
	}
	cout << "\nStarting cost "<<tempCost;
	costHistory.push_back(tempCost);

//Start training
//	size_t numTrainingLoops = 4000;
	size_t numTrainingLoops = 2000;
	for (size_t trainLoops = 0; trainLoops < numTrainingLoops; ++trainLoops) {
		for (size_t tSet = 0; tSet < data1.getNumSets()-numToSave; ++tSet) {
			testStruct.updateLayers(data1.getInputNodes(tSet));
			CNNStructure holdTempGradients(testCase);

// Fill holdTempGradients for this test set. 
			testStruct.makeGradPass(holdTempGradients, data1.getOutputNodes(tSet));


// Add the temp to the accumulator
			holdAccumGradients += holdTempGradients;
		}

// Divide by the number of entries. You may want to do other things to reduce it as well.
		holdAccumGradients.divideScaler(double(-gradientCutDown*data1.getNumSets()-numToSave));

// Modify the weights and biases.
		testStruct += holdAccumGradients;
// Calculate and store the new cost.To do this, sum up the cost across the entire data set. 
// Note, this is not really required for the training. It is so I can get a look at how the
// training progressed. I should see if I could get it cheaply as part of the calculation of 
// the gradient. For example, calculating a cost is not expensive if the layerNodes have
// already been updated. 

		double tempCost = 0.;
		for (size_t tSet = 0; tSet < data1.getNumSets()-numToSave; ++tSet) {
			tempCost+=testStruct.calcCost(data1.getInputNodes(tSet), data1.getOutputNodes(tSet))/double(data1.getNumSets()-numToSave);
		}
		costHistory.push_back(tempCost);
		++lapCounter;
//		if (lapCounter > 5)gradientCutDown = 10.;
//		if (lapCounter > 20)gradientCutDown = 5.;

		if (numSinceLastPrint > numBetweenPrints) {
			numSinceLastPrint = 0;
			cout << "[" << lapCounter << "] " << tempCost;
		}
		++numSinceLastPrint;
	}
//	cout << "\nFinal structure " << endl;
//	for (size_t iCnt = 0; iCnt < testStruct.getNumWeightsMatrices(); ++iCnt) {
//		cout << "\n Layer " << iCnt << endl;
//		testStruct.displayWeights(iCnt);
//	}

	cout << "\nCost history";
	cout << costHistory;
//Write the weights structure to file
	testStruct.writeToFile("./states/weightsFile5.txt");
// You should now have a trained network. Try it on some cases.
size_t countHits = 0, countMisses = 0;

for (size_t tSet = data1.getNumSets()-numToSave; tSet < data1.getNumSets(); ++tSet) {
		testStruct.updateLayers(data1.getInputNodes(tSet));

//		cout << "\n simpleTest " << data1.getInputNodes(tSet);
		cout << "\nsimpleNodes " << data1.getOutputNodes(tSet);
		cout << "Associated label " << data1.getLabels(tSet)<<endl;

		testStruct.displayLayerNodes(testStruct.getNumWeightsMatrices());

		double max = -10.;
		size_t indexToKeep = 0;
		for (size_t iCnt = 0; iCnt < data1.getOutputDimension();++iCnt) {
//Find the largest
			if (testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[iCnt] > max) {
				max = testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[iCnt];
				indexToKeep = iCnt;
			}
		}
		cout << "indexToKeep " << indexToKeep << " label " << data1.getLabels(tSet)<<"\n\n";
		if (indexToKeep == data1.getLabels(tSet)) {

			++countHits;
		}
		else
		{
			++countMisses;
			data1.displayImage(tSet);
		}
	}
	cout << "\nNumber of hits == " << countHits << " Number of misses = " << countMisses;

	return 0;
}