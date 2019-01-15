#include "CNNStructure.h"
#include <assert.h>
#include <iostream>
#include "myMathHelpers.h"
using namespace std;

CNNStructure::CNNStructure(vector<int> structure)
{
	assert(structure.size() > 1);
// Treating the biases like translations. I'm also going to assume that
// multiplication of the nodes by the resulting matrix has the nodes coming in as a 
// row vector. That implies that for each layer, its matrix will be N+1 rows x M columns where N is the
// number of nodes in the previous layer. 

	layers = structure;
	//Initialize the weights biases matrices.
	for (int layerCount= 0; layerCount < layers.size()-1; ++layerCount) {
		vector<vector<double>> tRows;
		for (int rowCount = 0; rowCount < layers[layerCount]+1; ++rowCount) {
			vector<double> tCols;
			for (int colCount = 0; colCount < layers[layerCount+1]; ++colCount) {
				if (rowCount != layers[layerCount]) {
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

cout<< "\nTest multi matrix initialization";
	for (int layerCount = 0; layerCount < weights.size(); ++layerCount) {
		cout << "\nLayerCount "<<layerCount<<weights[layerCount]<<"\n";
	}
// Combine the multiple layer matrices into a single transform.
// Put weights layer 0 into the new matrix. Add the extra column and row.
	for (int rowCount = 0; rowCount < weights[0].size(); ++rowCount) {
		vector<double>tempVec;
		for (int colCount = 0; colCount < weights[0][0].size(); ++colCount){ 
			tempVec.push_back(weights[0][rowCount][colCount]);
		}
		if (rowCount != weights[0].size() - 1) {
			tempVec.push_back(0);
		}
		else
		{
			tempVec.push_back(1);
		}
		singleMat.push_back(tempVec);
	}
	cout << "\nOriginal singleMat "<<singleMat;

	for (int layerCount = 0; layerCount < weights.size() - 1; ++layerCount) {
		for (int rowCount = 0; rowCount < weights[layerCount].size(); ++rowCount) {
			vector<double> tempVec;
			for (int colCount = 0; colCount < weights[layerCount][0].size(); ++colCount) {
				double tempSum = 0;
				for (int iCnt = 0; iCnt < weights[layerCount+1].size(); ++iCnt) {
					tempSum += singleMat[rowCount][iCnt] * weights[layerCount+1][iCnt][colCount];
				}
				tempVec.push_back(tempSum);
			}
			for (int colCount = 0; colCount < weights[layerCount][0].size(); ++colCount) {
				singleMat[rowCount][colCount] = tempVec[colCount];
			}
		}
	}
	cout << "\nFinished singleMat " << singleMat;
}

CNNStructure::~CNNStructure()
{
}

double CNNStructure::calcCost(vector<double> input, vector<double> desired) {
	cout << "\nMulti matrix method";
//It is assumed that the user supplies an input vector that only has the nodes and not
//the additional 1 that adds in the biases. 
	vector<double> nextNodes = input;
	nextNodes.push_back(1.);
	assert(nextNodes.size() == weights[0].size());
	for (int layerCount = 0; layerCount < weights.size(); ++layerCount) {
		vector<double> tempVec = vecMatMult(nextNodes, weights[layerCount]);
		cout << "\nNodes for layer " << layerCount << " " << tempVec;
		nextNodes.clear();
		nextNodes = tempVec;
		//Sigma function
		for (int iCnt = 0; iCnt < tempVec.size(); ++iCnt) {
			if (tempVec[iCnt] > 1.) {
				tempVec[iCnt] = 1.;
			}
			if (tempVec[iCnt] < 0.) {
				tempVec[iCnt] = 0.;
			}
		}
		nextNodes.push_back(1.);
	}
	double costSum = 0.;
	for (int iCnt = 0; iCnt < nextNodes.size()-1; ++iCnt) {
		costSum += pow((nextNodes[iCnt] - desired[iCnt]),2);
	}
	return(costSum);
}

double CNNStructure::getNumWeightsMatrices() {
	return(weights.size());
}
double CNNStructure::getNumWeightsRows(int layerNum) {
	return(weights[layerNum].size());
}
double CNNStructure::getNumWeightsCols(int layerNum) {
	return(weights[layerNum][0].size());
}