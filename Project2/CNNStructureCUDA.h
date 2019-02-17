#ifndef CNNSTRUCTURECUDA_H
#define CNNSTRUCTURECUDA_H
#include<vector>
using namespace std;
class CNNStructureCUDA
{
	vector<vector<vector<double>>> weights; //vector of weighting matrices
	vector<vector<double>> layerNodes;
	// At this point, the biases are kept in the weights matrix (similar to translation terms).

public:
	CNNStructureCUDA(const vector<int>& structure, double w = 0., double b = 0.);
	CNNStructureCUDA(const string& inFile);
	//	~CNNStructure();
	void addLayers(const vector<int>& structure);
	double calcCost(const vector<double>& input, const vector<double>& desired, const bool updateLayersBool = true);
	size_t getNumWeightsMatrices();
	size_t getNumWeightsRows(size_t layerNum);
	size_t getNumWeightsCols(size_t layerNum);
	void displayStructure();
	void displayWeights(const size_t& thisLayer);
	void displayLayerNodes(const size_t& thisLayer);
	vector<double> getLayerNodes(const size_t& thisLayer);
	CNNStructureCUDA& operator+=(const CNNStructureCUDA& rhs);
	void updateLayers(const vector<double>& input);
	void divideScaler(const double& factor);
	void makeGradPass(CNNStructureCUDA& tempGradStruct, const vector<double>& desired);
	void writeToFile(const string& fileName);
	vector<int> readFromFile(const string& inFileName);
	void setToZeros();
	vector<vector<vector<double>>>& getWeights() { return(weights); }

};

#endif
