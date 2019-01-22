#ifndef CNNSTRUCTURE_H
#define CNNSTRUCTURE_H
#include<vector>
using namespace std;
class CNNStructure
{
	vector<vector<vector<double>>> weights; //vector of weighting matrices
// At this point, the biases are kept in the weights matrix (similar to translation terms).
// Even though I currently keep layers, that information can be gleened from the weights matrix. 
// layers will likely go away. 
	vector<vector<double>> singleMat; //Combines the weighting matricies into one matrix.
public:
	CNNStructure(const vector<int>& structure);
	~CNNStructure();
	double calcCost(const vector<double>& input, const vector<double>& desired);
	vector < vector<double>> getSingleMat() { return (singleMat); }
	unsigned int getNumWeightsMatrices();
	unsigned int getNumWeightsRows(int layerNum);
	unsigned int getNumWeightsCols(int layerNum);
	void displayWeights(const int& thisLayer);
	CNNStructure& operator+=(const CNNStructure& rhs);

};

#endif