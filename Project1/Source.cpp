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
#include "time.h"
using namespace std;

void calcGradientParts(CNNStructure& holdAccumGradients,const vector<int> testCase,handNumberData& data,
						CNNStructure& holdTempGradients,CNNStructure testStruct,size_t begin, size_t end) {
//	cout << "\nInside calcGrad...";
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
	cout << "\nHardware concurrency "<<std::thread::hardware_concurrency();
	size_t numToSave;
	double gradientCutDown = 40.;
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
	string inFile = "./states/weightsFile9.txt";
//	string inFile = "./states/testWeights.txt";

	string outFile = "./states/weightsFile10.txtt";
	size_t numTrainingLoops = 1000;
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

vector<double> costHistory;
costHistory.reserve(numTrainingLoops);
//Get the starting cost.
auto beginCheck = chrono::high_resolution_clock::now();
	double tempCost = 0.;
	for (size_t tSet = 0; tSet < data1.getNumSets()-numToSave; ++tSet) {
		tempCost += testStruct.calcCost(data1.getInputNodes(tSet), data1.getOutputNodes(tSet)) / double(data1.getNumSets()-numToSave); 
	}
	auto endCheck = chrono::high_resolution_clock::now();
	cout << "\nTime to check cost " << chrono::duration<double, std::milli>(endCheck - beginCheck).count();
	cout << "\nStarting cost "<<tempCost;
	costHistory.push_back(tempCost);

//Start training
	size_t numThreads = std::thread::hardware_concurrency();
	vector<size_t> begin, end;
	begin.reserve(numThreads);
	end.reserve(numThreads);
//Create separate holdTempGradients(testCase) and holdAccumGradients(testCase) for each thread.
	vector<CNNStructure> holdAccumGradients(numThreads,CNNStructure(testCase));
	vector<CNNStructure> holdTempGradients(numThreads, CNNStructure(testCase));

	size_t totalSets = data1.getNumSets() - numToSave;
	size_t numForEachSplit = totalSets / numThreads;
	begin.push_back(0);
	end.push_back(numForEachSplit);
	for (size_t iCnt = 1; iCnt < numThreads; ++iCnt) {
		begin.push_back(end[iCnt-1]);
		end.push_back(begin[iCnt]+numForEachSplit);
	}
	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {
		cout << "\nBegin and end " << begin[iCnt] << " " << end[iCnt];
	}
	vector<thread> t;
	t.reserve(numThreads);
	for (size_t trainLoops = 0; trainLoops < numTrainingLoops; ++trainLoops) {
		auto beginLoopTime = chrono::high_resolution_clock::now();

//In the following, if I send the testStruct as a reference, I get a different answer for the 
//cost downstream. I can see where sending by value might speed things
//up, beause each thread would have its own copy. One possibility would be that by passing by
//reference, when one of the threads updates the layer nodes (note, the weights are not changed),
//it could change what another thread sees since they would all be sharing the same referenece.

		for (size_t tC = 0; tC < numThreads; ++tC) {
			t[tC] = thread(calcGradientParts, ref(holdAccumGradients[tC]), testCase, ref(data1),
				ref(holdTempGradients[tC]), testStruct, begin[tC], end[tC]);
		}
		for (size_t tC = 0; tC < numThreads; ++tC) {
			t[tC].join();
		}

		for (size_t tC = 1; tC < numThreads; ++tC) {
			holdAccumGradients[0] += holdAccumGradients[tC];
		}

// Divide by the number of entries. You may want to do other things to reduce it as well.
		holdAccumGradients[0].divideScaler(double(-gradientCutDown*(data1.getNumSets()-numToSave)));

// Modify the weights and biases.
		testStruct += holdAccumGradients[0];

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
			auto endLoopTime = chrono::high_resolution_clock::now();
			cout << " [" << lapCounter << "] cost " << tempCost<<" Time "<<
				chrono::duration<double, std::milli>(endLoopTime - beginLoopTime).count();

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
	for (unsigned int iCnt = 0; iCnt < 10; ++iCnt) {
		cout<<"\ncount " << iCnt << " total " << statTestTotal[iCnt]<<" and missed "<<statTestMissed[iCnt];
	}

	return 0;
}