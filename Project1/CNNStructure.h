#include<vector>
using namespace std;
class CNNStructure
{
	vector<int>layers; //Tells how many elements in each layer. layers' size tells how many layers.
	vector<vector<vector<double>>> weights; //vector of weighting matrices
// At this point, the biases are kept in the weights matrix (similar to translation terms).
// Even though I currently keep layers, that information can be gleened from the weights matrix. 
// layers will likely go away. 
public:
	CNNStructure(vector<int> structure);
	~CNNStructure();
	double calcCost(vector<double> input, vector<double> desired);
	double getNumWeightsMatrices();
	double getNumWeightsRows(int layerNum);
	double getNumWeightsCols(int layerNum);
};

