#ifndef CNNSTRUCTUREThrust_H
#define CNNSTRUCTUREThrust_H
#include<thrust\host_vector.h>
using namespace std;
using namespace thrust;
class CNNStructureThrust
{
	host_vector<host_vector<host_vector<double>>> weights; //host_vector of weighting matrices
	host_vector<host_vector<double>> layerNodes;
	// At this point, the biases are kept in the weights matrix (similar to translation terms).

public:
	CNNStructureThrust(const host_vector<int>& structure, double w = 0., double b = 0.);
	CNNStructureThrust(const string& inFile);
	//	~CNNStructure();
	void addLayers(const host_vector<int>& structure);
	double calcCost(const host_vector<double>& input, const host_vector<double>& desired, const bool updateLayersBool = true);
	size_t getNumWeightsMatrices();
	size_t getNumWeightsRows(size_t layerNum);
	size_t getNumWeightsCols(size_t layerNum);
	void displayStructure();
	void displayWeights(const size_t& thisLayer);
	void displayLayerNodes(const size_t& thisLayer);
	host_vector<double> getLayerNodes(const size_t& thisLayer);
	CNNStructureThrust& operator+=(const CNNStructureThrust& rhs);
	void updateLayers(const host_vector<double>& input);
	void divideScaler(const double& factor);
	void makeGradPass(CNNStructureThrust& tempGradStruct, const host_vector<double>& desired);
	void writeToFile(const string& fileName);
	host_vector<int> readFromFile(const string& inFileName);
	void setToZeros();
	void setWeights(size_t layer, size_t row, size_t col, double value = 0.);
	host_vector<host_vector<host_vector<double>>>& getWeights() { return(weights); }
	host_vector<host_vector<double>>& getLayerNodes() { return(layerNodes); }

};

#endif
