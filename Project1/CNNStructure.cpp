#include "CNNStructure.h"
#include <assert.h>
#include <iostream>
#include "myMathHelpers.h"
using namespace std;

CNNStructure::CNNStructure(const vector<int>& structure, double w, double b)
{
	assert(structure.size() > 1);
// Treating the biases like translations. I'm also going to assume that
// multiplication of the nodes by the resulting matrix has the nodes coming in as a column
// vector on the right side. That implies that for each layer, its matrix will be M+1 rows x N+1
//	columns (the additional column being 1s and the additional row being 0s with a trailing 1) 
// where M is the number of nodes in the resulting layer
// and N is the number of nodes in the input layer, before adding the column of ones at the end. 

	//Initialize the weights and biases matrices. Initialize the size of layerNodes at the same time.
	for (int layerCount= 0; layerCount < structure.size()-1; ++layerCount) {
		vector<vector<double>> tRows;
		for (int rowCount = 0; rowCount < structure[layerCount+1]+1; ++rowCount) {
			vector<double> tCols;
			for (int colCount = 0; colCount < structure[layerCount]+1; ++colCount) {
				if (colCount != structure[layerCount]) {
					tCols.push_back(w);
				}
				else
				{
					tCols.push_back(b);
				}
			}
			if (rowCount == structure[layerCount + 1]) {
				for (size_t iCnt = 0; iCnt < structure[layerCount]; ++iCnt) {
					tCols[iCnt] = 0.;
				}
				tCols[structure[layerCount] ] = 1.;
			}
				tRows.push_back(tCols);
		} 
		weights.push_back(tRows);
	}
	for (int layerCount = 0; layerCount < structure.size(); ++layerCount) {
		vector<double> tempVec;
		for (int colCount = 0; colCount < structure[layerCount]; ++colCount) {
			tempVec.push_back(0.);
		}
		tempVec.push_back(1.);
		layerNodes.push_back(tempVec);
	}
}

/*
CNNStructure::~CNNStructure()
{
}
*/

double CNNStructure::calcCost(const vector<double>& input, const vector<double>& desired) {
//This version uses the input to update the layers and then calculate the cost.
	updateLayers(input);
//The cost only depends on the last layer's nodes. And there is no addition term added to the vector.
	double costSum = 0.;
	size_t numLayers = layerNodes.size();
	for (int iCnt = 0; iCnt < desired.size()-1; ++iCnt) {	// Cut down by one because desired has the extra 1.
															// It doesn't really matter since both have 1 at the end.
		costSum += pow((layerNodes[numLayers-1][iCnt] - desired[iCnt]),2);
	}
	return(costSum);
}

size_t CNNStructure::getNumWeightsMatrices() {
	return(weights.size());
}
size_t CNNStructure::getNumWeightsRows(const size_t layerNum) {
	return(weights[layerNum].size());
}
size_t  CNNStructure::getNumWeightsCols(const size_t layerNum) {
	return(weights[layerNum][0].size());
}
vector<double> CNNStructure::getLayerNodes(const size_t& thisLayer) {
	return(layerNodes[thisLayer]);
}

void CNNStructure::displayWeights(const size_t& thisLayer) {
	cout << weights[thisLayer];
}
void CNNStructure::displayStructure() {
	cout << weights;
}

void CNNStructure::displayLayerNodes(const size_t& thisLayer) {
	cout << layerNodes[thisLayer];
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
//The m,N element is reserved to be 1.
		weights[layerCnt][weights[layerCnt].size() - 1][weights[layerCnt][0].size() - 1] = 1.;
	}

	return *this;
}

void CNNStructure::updateLayers(const vector<double>& input) {
//input contains the starting layer nodes. 
	layerNodes[0] = input;
	vector<double> tempLayer;
	for (int layerCount = 0; layerCount < weights.size(); ++layerCount) {
		if (layerCount == 0) {
			tempLayer = input;
		}
		assert(tempLayer.size() == weights[layerCount][0].size());
		vector<double> tempVec = matVecMult(weights[layerCount], tempLayer);
//Sigma function
		for (int iCnt = 0; iCnt < tempVec.size(); ++iCnt) {

			if (tempVec[iCnt] < 0.) {
				tempVec[iCnt] = 0.;			//Comment it to Kill it till you understand it.
			}
		}
		layerNodes[layerCount+1] = tempVec;
		tempLayer = tempVec;	
	}
}

void CNNStructure::divideScaler(const double& factor) {
	assert(factor != 0.);
	for (size_t layerCount = 0; layerCount < weights.size(); ++layerCount) {
		for (size_t rowCount = 0; rowCount < weights[layerCount].size(); ++rowCount) {
			for (size_t colCount = 0; colCount < weights[layerCount][0].size(); ++colCount) {
				weights[layerCount][rowCount][colCount] /= factor;
			}
		}
	}
}

void CNNStructure::makeGradPass(CNNStructure& tempGradStruct,const vector<double>& desired) {
// The goal here is to create the gradient for the single test case. 
// There are multiple terms that need to be multiplied
// together to form each element. I believe that I need to complete a layer (from back to front) 
// before proceding to the next layer. The reason is that you need the results of layer L
// inorder to get a cost for L-1.
	for (size_t layerCount = weights.size(); layerCount > 0; --layerCount) {
		vector<double> z = matVecMult(weights[layerCount-1], layerNodes[layerCount-1]);
		vector<double> pCpA = layerNodes[layerCount]-desired;
		pCpA = 2.*pCpA;
		for (size_t rowCount = 0; rowCount < weights[layerCount-1].size()-1; ++rowCount) {
			if (z[rowCount] < 0.) {
				z[rowCount] = 0.;
			}
			else {
				z[rowCount] = 1.;
			}
//			z[rowCount] = 1.;	//uncomment here and comment above to Kill sigma till you understand it.
			for (size_t colCount = 0; colCount < weights[layerCount-1].size(); ++colCount) {
//(partial z wrt w)*partial relu*pCpA
				tempGradStruct.weights[layerCount-1][rowCount][colCount] = 
					layerNodes[layerCount-1][colCount]*z[rowCount]*pCpA[rowCount];
			}
// Each row also has a bias term at the end of the row.
			tempGradStruct.weights[layerCount - 1][rowCount][weights[layerCount - 1].size() ] =
				z[rowCount] * pCpA[rowCount];

		}
	}
}
