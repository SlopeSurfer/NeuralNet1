#include "CNNStructureThrust.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <sstream>
#include "myMathHelpersThrust.h"
#include <string>
//using namespace std;
using namespace thrust;

CNNStructureThrust::CNNStructureThrust(const host_vector<int>& structure, double w, double b)
{
	assert(structure.size() > 1);
	// Treating the biases like translations. I'm also going to assume that
	// multiplication of the nodes by the resulting matrix has the nodes coming in as a column
	// host_vector on the right side. That implies that for each layer, its matrix will be M rows x N
	//	columns.
	// where M is the number of rows in the resulting layer
	// and N is the number of columns. 

		//Initialize the weights and biases matrices. Initialize the size of layerNodes at the same time.
	for (int layerCount = 0; layerCount < structure.size() - 1; ++layerCount) {
		host_vector<host_vector<double>> tRows;
		tRows.reserve(structure[layerCount + 1] - 1);
		for (int rowCount = 0; rowCount < structure[layerCount + 1] - 1; ++rowCount) {
			host_vector<double> tCols;
			tCols.reserve(structure[layerCount]);
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
		host_vector<double> tCols;
		tCols.reserve(structure[layerCount] - 1);
		//Add the row of 0s ending in 1.
		for (size_t iCnt = 0; iCnt < structure[layerCount] - 1; ++iCnt) {
			tCols.push_back(0.);
		}
		tCols.push_back(1.);
		tRows.push_back(tCols);
		weights.push_back(tRows);
	}
	//Create the layers that go with these matrices.
	addLayers(structure);
}

CNNStructureThrust::CNNStructureThrust(const string& inFile) {
	host_vector<int> source = readFromFile(inFile);
	addLayers(source);
}

void CNNStructureThrust::addLayers(const host_vector<int>& structure) {
	//Create space for the layer nodes.
	assert(layerNodes.size() == 0);
	for (int layerCount = 0; layerCount < structure.size(); ++layerCount) {
		host_vector<double> tempVec;
		tempVec.reserve(structure[layerCount]);
		for (int colCount = 0; colCount < structure[layerCount] - 1; ++colCount) {
			tempVec.push_back(0.);
		}
		tempVec.push_back(1.);
		layerNodes.push_back(tempVec);
	}
}

double CNNStructureThrust::calcCost(const host_vector<double>& input, const host_vector<double>& desired, const bool updateLayersBool) {
	//updateLayersBool true by default.
	if (updateLayersBool) {
		updateLayers(input);
	}
	//The cost only depends on the last layer's nodes. And there is no addition term added to the host_vector.
	double costSum = 0.;
	size_t numLayers = layerNodes.size();
	for (int iCnt = 0; iCnt < desired.size() - 1; ++iCnt) {	// Cut down by one because desired has the extra 1.
															// It doesn't really matter since both have 1 at the end.
		costSum += pow((layerNodes[numLayers - 1][iCnt] - desired[iCnt]), 2);
	}
	return(costSum);
}

size_t CNNStructureThrust::getNumWeightsMatrices() {
	return(weights.size());
}
size_t CNNStructureThrust::getNumWeightsRows(const size_t layerNum) {
	return(weights[layerNum].size());
}
size_t  CNNStructureThrust::getNumWeightsCols(const size_t layerNum) {
	return(weights[layerNum][0].size());
}
host_vector<double> CNNStructureThrust::getLayerNodes(const size_t& thisLayer) {
	return(layerNodes[thisLayer]);
}

void CNNStructureThrust::displayWeights(const size_t& thisLayer) {
	cout << weights[thisLayer];
}
void CNNStructureThrust::displayStructure() {
	cout << weights;
}

void CNNStructureThrust::displayLayerNodes(const size_t& thisLayer) {
	cout << layerNodes[thisLayer];
}

CNNStructureThrust & CNNStructureThrust::operator+=(const CNNStructureThrust& rhs) {
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

void CNNStructureThrust::updateLayers(const host_vector<double>& input) {
	//input contains the starting layer nodes. 
	layerNodes[0] = input;
	host_vector<double> tempLayer = input;	//Look into making presized vectors to hold this
	for (int layerCount = 0; layerCount < weights.size(); ++layerCount) {
		assert(tempLayer.size() == weights[layerCount][0].size());
		host_vector<double> tempVec = matVecMult(weights[layerCount], tempLayer);
		//Sigma function
		for (int iCnt = 0; iCnt < tempVec.size(); ++iCnt) {

			if (tempVec[iCnt] < 0.) {
				tempVec[iCnt] = 0.;			//Comment it if you want to kill sigma.
			}
		}
		layerNodes[layerCount + 1] = tempVec;
		tempLayer.resize(tempVec.size());	//Before I added this, I think it only worked because the sizes never got bigger (785, 16, 16, 11)
		tempLayer = tempVec;
	}
}

void CNNStructureThrust::divideScaler(const double& factor) {
	assert(factor != 0.);
	for (size_t layerCount = 0; layerCount < weights.size(); ++layerCount) {
		for (size_t rowCount = 0; rowCount < weights[layerCount].size(); ++rowCount) {
			for (size_t colCount = 0; colCount < weights[layerCount][0].size(); ++colCount) {
				weights[layerCount][rowCount][colCount] /= factor;
			}
		}
	}
}

void CNNStructureThrust::makeGradPass(CNNStructureThrust& tempGradStruct, const host_vector<double>& desired) {
	// The goal here is to create the gradient for the single test case. 
	// There are multiple terms that need to be multiplied
	// together to form each element. Complete a layer (from back to front) 
	// before proceding to the next layer. The reason is that you need the results of layer L
	// inorder to get a cost for L-1.

	host_vector<double> pCpA = 2 * (layerNodes[weights.size()] - desired); //Overloaded mult and minus.;


	for (size_t layerCount = weights.size(); layerCount > 0; --layerCount) {
		host_vector<double> partRelu = matVecMult(weights[layerCount - 1], layerNodes[layerCount - 1]);
	
		//Sigma
		for (size_t rowCount = 0; rowCount < weights[layerCount - 1].size() - 1; ++rowCount) {

			if (partRelu[rowCount] < 0.) {
				partRelu[rowCount] = 0.;
			}
			else {
				partRelu[rowCount] = 1.;
			}

			//			partRelu[rowCount] = 1.;	//uncomment here and comment above to Kill sigma till you understand it.
			for (size_t colCount = 0; colCount < weights[layerCount - 1][0].size() - 1; ++colCount) {
				//(partial z wrt w)*partial relu*pCpA
				tempGradStruct.weights[layerCount - 1][rowCount][colCount] +=
					layerNodes[layerCount - 1][colCount] * partRelu[rowCount] * pCpA[rowCount];
			}
			// Each row also has a bias term at the end of the row.
			tempGradStruct.weights[layerCount - 1][rowCount][weights[layerCount - 1][0].size() - 1] +=
				partRelu[rowCount] * pCpA[rowCount];
			cout << "\n tempGradStruct " << tempGradStruct.weights[layerCount - 1][rowCount][weights[layerCount - 1][0].size() - 1]
					<< " " << rowCount<<" "<<layerCount-1;
		}
		if (layerCount > 1) {
			host_vector<double> temppCpA;
			temppCpA.reserve(weights[layerCount - 1][0].size());
			//Calculate the pCpA host_vector for the next round.
			for (size_t colCount = 0; colCount < weights[layerCount - 1][0].size() - 1; ++colCount) {
				double tempSum = 0.;
				for (size_t rowCount = 0; rowCount < weights[layerCount - 1].size() - 1; ++rowCount) {
					tempSum += weights[layerCount - 1][rowCount][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}
				temppCpA.push_back(tempSum);
			}
	
			pCpA = temppCpA;
		}
	}
}

void CNNStructureThrust::writeToFile(const string& outFileName) {
	ofstream outFile(outFileName);
	//The header is the size of each set of nodes corresponding to each layer.

	if (outFile.is_open()) {
		outFile << " " << weights[0][0].size();
		for (size_t iCnt = 0; iCnt < weights.size(); ++iCnt) {
			outFile << " " << weights[iCnt].size();
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

host_vector<int> CNNStructureThrust::readFromFile(const string& inFileName) {
	ifstream inFile(inFileName);
	//The header is the number of layers, and then the size of each set of nodes
	//corresponding to each layer.

	if (inFile.is_open()) {
		//Get the first line and fill the structure host_vector.
		weights.clear();
		string line;
		getline(inFile, line);
		istringstream in(line);
		host_vector<int> structure = host_vector<int>(istream_iterator<double>(in), istream_iterator<double>());
		for (size_t matrixCount = 0; matrixCount < structure.size() - 1; ++matrixCount) {
			host_vector<host_vector<double>>tempMat;
			for (size_t rowCount = 0; rowCount < structure[matrixCount + 1]; ++rowCount) {
				getline(inFile, line);
				//Strip the [ and ] and replace the commas with spaces.
				line.erase(remove(line.begin(), line.end(), '['), line.end());	//cannot replace with ''
				line.erase(remove(line.begin(), line.end(), ']'), line.end());
				thrust::replace(line.begin(), line.end(), ',', ' ');

				if (!line.empty()) {
					istringstream in(line);
					host_vector<double> tempVec = host_vector<double>(istream_iterator<double>(in), istream_iterator<double>());
					tempMat.push_back(tempVec);
				}
				else {
					//The above might let some empty lines slip through. Ignore them and don't let them count toward your rows. 
					//					cout << "\nEmpty string ignored\n";
					--rowCount;
				}
			}
			if (!tempMat.empty()) {
				weights.push_back(tempMat);
			}
		}

		inFile.close();
		return(structure);
	}
	else
	{
		cout << "Could not open file " << inFileName << " for reading";
		host_vector<int> temp;
		temp.push_back(-1);
		return(temp);
	}
}

void CNNStructureThrust::setToZeros() {

	for (size_t layerCount = 0; layerCount < this->weights.size(); ++layerCount) {
		host_vector<double> vecTemp(this->weights[layerCount][0].size(), 0);
		for (size_t rowCount = 0; rowCount < this->weights[layerCount].size(); ++rowCount) {
			this->weights[layerCount][rowCount] = vecTemp;
		}

	}

}
void CNNStructureThrust::setWeights(size_t layer, size_t row, size_t col, double value) {
	weights[layer][row][col] = value;
}