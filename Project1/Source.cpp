#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include <thread>
#include "imagesAndLabels.h"
#include "CNNStructure.h"
#include "myMathHelpers.h"
#include "dataSet.h"
#include "handNumberData.h"
#include <unordered_map>
#include "limits.h"
using namespace std;

void calcGradientParts(CNNStructure& holdAccumGradients,const vector<int>& testCase,handNumberData& data,
						CNNStructure& holdTempGradients,CNNStructure testStruct,size_t begin, size_t end) {
	holdAccumGradients.setToZeros();	//Zero this out because of the += later.
	for (size_t tSet = begin; tSet < end; ++tSet) {
		testStruct.updateLayers(data.getInputNodes(tSet));

		// Fill holdTempGradients for this test set. 
		testStruct.makeGradPass(holdTempGradients, data.getOutputNodes(tSet));

		// Add the temp to the accumulator
		holdAccumGradients += holdTempGradients;
	}
}

int main() {

//Set up the training data.
	size_t numToSave;
	double gradientCutDown = 40.;
	size_t lapCounter = 0,numBetweenPrints = 9,numSinceLastPrint = 0;
	// Set up a test case for the structure
	vector<int> testCase;

	string fileNameLabels = "./data/train-labels.idx1-ubyte";
	string fileNameImages = "./data/train-images.idx3-ubyte";

	handNumberData data1(fileNameImages, fileNameLabels);
	data1.displayImage(5);
	testCase.push_back((int)data1.getInputDimension());
	testCase.push_back(16);
	testCase.push_back(16);
	testCase.push_back((int)data1.getOutputDimension() );

	numToSave = 0;
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
	string inFile = "./states/weightsRound2-24.txt";
//	string inFile = "./states/testWeights.txt";

	string outFile = "./states/weightsRound2-25.txt";
	size_t numTrainingLoops = 100;
	CNNStructure testStruct(inFile);

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

//I will need a separate holdTempGradients(testCase) and holdAccumGradients for each thread.
	CNNStructure holdAccumGradients1(testCase);
	CNNStructure holdAccumGradients2(testCase);
	CNNStructure holdAccumGradients3(testCase);
	CNNStructure holdAccumGradients4(testCase);
	CNNStructure holdTempGradients1(testCase); //Each member will get set with no depedence on previous.
	CNNStructure holdTempGradients2(testCase); //Each member will get set with no depedence on previous.
	CNNStructure holdTempGradients3(testCase); //Each member will get set with no depedence on previous.
	CNNStructure holdTempGradients4(testCase); //Each member will get set with no depedence on previous.
	size_t totalSets = data1.getNumSets() - numToSave;
	size_t numForEachSplit = totalSets / 4;
	size_t begin1 = 0, end1 = numForEachSplit;
	size_t begin2 = end1 + 1, end2 = end1 + numForEachSplit;
	size_t begin3 = end2 + 1, end3 = end2 + numForEachSplit;
	size_t begin4 = end3 + 1, end4 = end3 + numForEachSplit;
	for (size_t trainLoops = 0; trainLoops < numTrainingLoops; ++trainLoops) {

		thread t1 = thread(calcGradientParts, ref(holdAccumGradients1), ref(testCase), ref(data1), 
			ref(holdTempGradients1), testStruct, begin1, end1);
		thread t2 = thread(calcGradientParts, ref(holdAccumGradients2), ref(testCase), ref(data1),
			ref(holdTempGradients2), testStruct, begin2, end2);
		thread t3 = thread(calcGradientParts, ref(holdAccumGradients3), ref(testCase), ref(data1),
			ref(holdTempGradients3), testStruct, begin3, end3);
		thread t4 = thread(calcGradientParts, ref(holdAccumGradients4), ref(testCase), ref(data1),
			ref(holdTempGradients4), testStruct, begin4, end4);

		t1.join();
		t2.join();
		t3.join();
		t4.join();

		holdAccumGradients1 += holdAccumGradients2;
		holdAccumGradients1 += holdAccumGradients3;
		holdAccumGradients1 += holdAccumGradients4;

// Divide by the number of entries. You may want to do other things to reduce it as well.
		holdAccumGradients1.divideScaler(double(-gradientCutDown*(data1.getNumSets()-numToSave)));

// Modify the weights and biases.
		testStruct += holdAccumGradients1;

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
//		if (lapCounter > 5)gradientCutDown = 20.;
//		if (lapCounter > 200)gradientCutDown = 10.;

		if (numSinceLastPrint > numBetweenPrints) {
			numSinceLastPrint = 0;
			cout << "[" << lapCounter << "] " << tempCost;
		}
		++numSinceLastPrint;
	}

	cout << "\nCost history";
	cout << costHistory;
//Write the weights structure to file
	testStruct.writeToFile(outFile);
// Try it on some cases.
size_t countHits = 0, countMisses = 0;

unordered_map<int, int> statTestTotal = { {0,0},{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0} };
unordered_map<int, int> statTestMissed = { {0,0},{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0} };

//for (size_t tSet = data1.getNumSets()-numToSave; tSet < data1.getNumSets(); ++tSet) {
	for (size_t tSet = 0; tSet < 200; ++tSet) {
		statTestTotal[data1.getLabel(tSet)]++;
		testStruct.updateLayers(data1.getInputNodes(tSet));

		testStruct.displayLayerNodes(testStruct.getNumWeightsMatrices());

		double max = -DBL_MAX;
		size_t indexToKeep = 0;
		for (size_t iCnt = 0; iCnt < data1.getOutputDimension()-1;++iCnt) {
//Find the largest
			if (testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[iCnt] > max) {
				max = testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[iCnt];
				indexToKeep = iCnt;
			}
		}
		cout << "Chosen index " << indexToKeep << " label " << data1.getLabel(tSet)<<"\n\n";
		if (indexToKeep == data1.getLabel(tSet)) {
			++countHits;
		}
		else
		{
			++countMisses;
			data1.displayImage(tSet);
			statTestMissed[data1.getLabel(tSet)]++;
		}
	}
	cout << "\nNumber of hits == " << countHits << " Number of misses = " << countMisses;
	for (size_t iCnt = 0; iCnt < 10; ++iCnt) {
		cout<<"\ncount " << iCnt << " total " << statTestTotal[iCnt]<<" ane missed "<<statTestMissed[iCnt];
	}

	return 0;
}