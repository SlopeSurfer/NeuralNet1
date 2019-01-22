#include "CNNStructure.h"
#include <assert.h>
#include <iostream>
#include "myMathHelpers.h"
using namespace std;

CNNStructure::CNNStructure(const vector<int>& structure)
{
	assert(structure.size() > 1);
// Treating the biases like translations. I'm also going to assume that
// multiplication of the nodes by the resulting matrix has the nodes coming in as a 
// row vector. That implies that for each layer, its matrix will be N+1 rows x M columns where N is the
// number of nodes in the previous layer. 

	//Initialize the weights biases matrices.
	for (int layerCount= 0; layerCount < structure.size()-1; ++layerCount) {
		vector<vector<double>> tRows;
		for (int rowCount = 0; rowCount < structure[layerCount]+1; ++rowCount) {
			vector<double> tCols;
			for (int colCount = 0; colCount < structure[layerCount+1]; ++colCount) {
				if (rowCount != structure[layerCount]) {
					tCols.push_back(.5);
				}
				else
				{
					tCols.push_back(.1);
				}
			}
			tRows.push_back(tCols);
		} 
//Add the bias here.
		
		weights.push_back(tRows);
	}
}

CNNStructure::~CNNStructure()
{
}

double CNNStructure::calcCost(const vector<double>& input, const vector<double>& desired) {
	cout << "\nMulti matrix method";
//It is assumed that the user supplies an input vector that only has the nodes and not
//the additional 1 that adds in the biases. 
	vector<double> nextNodes = input;

	for (int layerCount = 0; layerCount < weights.size(); ++layerCount) {
		nextNodes.push_back(1.);	// Adds bias as a translation.
		assert(nextNodes.size() == weights[layerCount].size());		vector<double> tempVec = vecMatMult(nextNodes, weights[layerCount]);
		//Sigma function
		for (int iCnt = 0; iCnt < tempVec.size(); ++iCnt) {
//			if (tempVec[iCnt] > 1.) {
//				tempVec[iCnt] = 1.;
//			}
			if (tempVec[iCnt] < 0.) {
				tempVec[iCnt] = 0.;
			}
		}
		cout << "\nNodes for layer " << layerCount << " " << tempVec;

		nextNodes.clear();	//Not sure if this is necessary.
		nextNodes = tempVec;
	}
//The cost only depends on the last layer's nodes. And there is not addition term added to the vector.
	double costSum = 0.;
	for (int iCnt = 0; iCnt < nextNodes.size(); ++iCnt) {
		costSum += pow((nextNodes[iCnt] - desired[iCnt]),2);
	}
	return(costSum);
}

unsigned int CNNStructure::getNumWeightsMatrices() {
	return(weights.size());
}
unsigned int CNNStructure::getNumWeightsRows(int layerNum) {
	return(weights[layerNum].size());
}
unsigned int CNNStructure::getNumWeightsCols(int layerNum) {
	return(weights[layerNum][0].size());

}

void CNNStructure::displayWeights(const int& thisLayer) {
	cout << weights[thisLayer];
}

CNNStructure & CNNStructure::operator+=(const CNNStructure& rhs) {
	assert(weights.size() == rhs.weights.size());
	assert(weights[0].size() == rhs.weights[0].size());
	assert(weights[0][0].size() == rhs.weights[0][0].size());
	for (size_t layerCnt = 0; layerCnt < weights.size(); ++layerCnt) {
		for (size_t rowCnt = 0; rowCnt < weights[layerCnt].size(); ++rowCnt) {
			for (size_t colCnt = 0; colCnt < weights[layerCnt][0].size(); ++colCnt) {
				weights[layerCnt][rowCnt][colCnt] = weights[layerCnt][rowCnt][colCnt] +
					rhs.weights[layerCnt][rowCnt][colCnt];
			}
		}
	}
	return *this;
}