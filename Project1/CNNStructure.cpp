#include "CNNStructure.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <sstream>
#include "myMathHelpers.h"
#include <string>
using namespace std;

CNNStructure::CNNStructure(const vector<int>& structure, double w, double b)
{
	assert(structure.size() > 1);
// Treating the biases like translations. I'm also going to assume that
// multiplication of the nodes by the resulting matrix has the nodes coming in as a column
// vector on the right side. That implies that for each layer, its matrix will be M rows x N
//	columns.
// where M is the number of rows in the resulting layer
// and N is the number of columns. 

	//Initialize the weights and biases matrices. Initialize the size of layerNodes at the same time.
	for (int layerCount= 0; layerCount < structure.size()-1; ++layerCount) {
		vector<vector<double>> tRows;
		for (int rowCount = 0; rowCount < structure[layerCount+1]-1; ++rowCount) {
			vector<double> tCols;
			for (int colCount = 0; colCount < structure[layerCount]; ++colCount) {
				if (colCount != structure[layerCount]) {
					if (w != 0.) {
						tCols.push_back(float(rand() % 100) / 100.);
					}
					else
					{
						tCols.push_back(w);
					}
				}
				else
				{
					if (b != 0) {
						tCols.push_back(float(rand() % 100) / 100.);
					}
					else
					{
						tCols.push_back(b);

					}
				}
			}		
			tRows.push_back(tCols);
		} 
vector<double> tCols;
//Add the row of 0s ending in 1.
		for (size_t iCnt = 0; iCnt < structure[layerCount]-1; ++iCnt) {
			tCols.push_back(0.);
		}
		tCols.push_back(1.);
		tRows.push_back(tCols);
		weights.push_back(tRows);
	}
//Create the layers that go with these matrices.
	addLayers(structure);
}

CNNStructure::CNNStructure(const string& inFile) {
	vector<int> source = readFromFile(inFile);
	cout << "\nSource back from readfile " << source;	
	addLayers(source);
}

void CNNStructure::addLayers(const vector<int>& structure) {
	for (int layerCount = 0; layerCount < structure.size(); ++layerCount) {
		vector<double> tempVec;
		for (int colCount = 0; colCount < structure[layerCount] - 1; ++colCount) {
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
	vector<double> pCpA;
	for (size_t layerCount = weights.size(); layerCount > 0; --layerCount) {
		vector<double> partRelu = matVecMult(weights[layerCount - 1], layerNodes[layerCount - 1]);
		if (layerCount == weights.size()) {
			pCpA = 2 * (layerNodes[layerCount] - desired); //Overloaded mult and minus.
		}
//Sigma
		for (size_t rowCount = 0; rowCount < weights[layerCount - 1].size() - 1; ++rowCount) {

			if (partRelu[rowCount] < 0.) {
				partRelu[rowCount] = 0.;
			}
			else {
				partRelu[rowCount] = 1.;
			}
//			partRelu[rowCount] = 1.;	//uncomment here and comment above to Kill sigma till you understand it.
//			cout<<"\nweights[layerCount - 1][0].size() " << weights[layerCount - 1][0].size();
			for (size_t colCount = 0; colCount < weights[layerCount - 1][0].size()-1; ++colCount) {
//(partial z wrt w)*partial relu*pCpA
				tempGradStruct.weights[layerCount - 1][rowCount][colCount] =
					layerNodes[layerCount - 1][colCount] * partRelu[rowCount] *pCpA[rowCount];
			}
// Each row also has a bias term at the end of the row.
			tempGradStruct.weights[layerCount - 1][rowCount][weights[layerCount - 1][0].size()-1] =
				partRelu[rowCount] * pCpA[rowCount];
		}

//		if (layerCount > 1) {
			vector<double> temppCpA;
			//Calculate the pCpA vector for the next round.
			for (size_t colCount = 0; colCount < weights[layerCount - 1][0].size()-1; ++colCount) {
				double tempSum = 0.;
				for (size_t rowCount = 0; rowCount < weights[layerCount - 1].size() - 1; ++rowCount) {
					tempSum += weights[layerCount - 1][rowCount][colCount]*partRelu[rowCount]*pCpA[rowCount];
				}	
				temppCpA.push_back(tempSum);
			}
			pCpA.clear();
			pCpA =temppCpA;
		}
//	}
}

void CNNStructure::writeToFile(const string& outFileName) {
	ofstream outFile(outFileName);
//The header will be the number of layers, and then the size of each set of nodes
//corresponding to each layer.

	if (outFile.is_open()) {
//		outFile << weights.size();
		outFile << " "<<weights[0][0].size();
		for (size_t iCnt = 0; iCnt < weights.size(); ++iCnt) {
			outFile <<" "<< weights[iCnt].size();
		}
		outFile << "\n";
		outFile << weights;
		outFile.close();
	}
	else
	{
		cout << "\nCould not open file " << outFileName << " for writing";
	}
}

vector<int> CNNStructure::readFromFile(const string& inFileName) {
	ifstream inFile(inFileName);
	//The header is the number of layers, and then the size of each set of nodes
	//corresponding to each layer.
	
	if (inFile.is_open()) {
//Get the first line and fill the structure vector.
		weights.clear();
		string line; getline(inFile, line);
		istringstream in(line);
		vector<int> structure = vector<int>(istream_iterator<double>(in), istream_iterator<double>());
		for (size_t matrixCount = 0; matrixCount < structure.size(); ++matrixCount) {
			vector<vector<double>>tempMat;
			for (size_t rowCount = 0; rowCount < structure[matrixCount + 1]; ++rowCount) {
				getline(inFile, line);
//Strip the [ and ] and replace the commas with spaces.
				line.erase(remove(line.begin(), line.end(), '['), line.end());	//cannot replace with ''
				line.erase(remove(line.begin(), line.end(), ']'), line.end());
				replace(line.begin(), line.end(), ',', ' ');

				if (!line.empty()) {
					istringstream in(line);
					vector<double> tempVec = vector<double>(istream_iterator<double>(in), istream_iterator<double>());
					tempMat.push_back(tempVec);
				}
				else{
//The above let's some empty lines slip through. Ignore them and don't let them count toward you rows. 
//					cout << "\nEmpty string ignored\n";
					--rowCount;
				}
			}
			if (!tempMat.empty()) {
//				cout << "Got Here";
				weights.push_back(tempMat);
			}
		}

		inFile.close();
		cout << "\nReturning structure from read " << structure;
		return(structure);
	}
	else
	{
		cout << "Could not open file " << inFileName << " for reading";
		vector<int> temp;
		temp.push_back(-1);
		return(temp);
	}

}

