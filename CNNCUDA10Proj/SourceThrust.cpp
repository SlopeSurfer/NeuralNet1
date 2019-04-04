#include <iostream>
#include <fstream>
#include <sstream>
#include <thrust\host_vector.h>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include "imagesAndLabelsThrust.h"
#include "CNNStructureThrust.h"
#include "myMathHelpersthrust.h"
#include "dataSetThrust.h"
#include "handNumberDataThrust.h"
#include <unordered_map>
#include "limits.h"
#include <chrono>
#include <stdlib.h>
//#include <stdio.h>
using namespace std;
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA

#include <helper_cuda.h>
#include "CStructs.h"

//Next step for conversion to CUDA
//Make a float version of calcGradientPartsC. As you change its input from C to CFlat
//do the work with CUDA Kernels. Start with holdAccumGradients. Note that you could use the 
//Flattened types to vary the signature of calls like setWeightsToZero. So, if its flat, calls Kernel version.
//But, chances are that the whole thing is different enough that you'll really have different version of 
//calcGradientPartsC.

extern "C" void launchCuda(int* a, int* b, int*c, int n);//
extern "C" void launchCudaMatrix(size_t numLoops, SimpleMatrix** myMatrix1, SimpleMatrix** myMatrix2);
extern "C" void launchCudaPitch(int width, int height, size_t pitch, int* d_tab);
extern "C" void launchCudaMatVecMult(double *d_cc, double *d_varArray, double* gatherArray,
	dim3 blockSize, dim3 gridSize, int nx, int ny, int nz);
extern "C" void launchCudaPVecReduce(double *d_constMatrix, double* d_varArray, double* d_gatherArray,
	dim3 blockSize, dim3 gridSize, int nx, int ny, int nz, double*outArray, bool shared = true);


/*
void launchCudaMatrix(size_t numLoops, SimpleMatrix** myMatrix1, SimpleMatrix** myMatrix2) {
	//void launchCudaMatrix(size_t numLoops, int** myMatrix1, int** myMatrix2) {
	printf("\nnumLoops in launchCudaMatrix %d ", numLoops);

	size_t numRows = myMatrix1[0]->numRows;
	size_t numCols = myMatrix1[0]->numCols;
	for (size_t loops = 0; loops < numLoops; ++loops) {

		for (size_t row = 0; row < numRows; ++row) {
			for (size_t col = 0; col < numCols; ++col) {
				myMatrix1[loops]->matrix[row*numCols + col] = myMatrix1[loops]->matrix[row*numCols + col] +
					myMatrix2[loops]->matrix[row*numCols + col];
				//				myMatrix1->setMatrixElem(row, col, myMatrix1->getMatrixElem(row, col) + myMatrix2->getMatrixElem(row, col));
			}
		}
	}

	cudaDeviceSynchronize();

}
*/

//extern "C" void makeGradPassC(CNNStructureC* testStruct, CNNStructureC* tempGradStruct, double* desired);

extern "C" void calcGradientPartsC(CNNStructureC *holdAccumGradients, structureC<int> *testCase, dataC* data,
	CNNStructureC *testStruct, size_t begin, double* cost, reducedLayerNodesC* nodesForUpdating, 
	const size_t numBlocks, const size_t numThreads);

extern "C" void calcGradientPartsCUDA(CNNStructureCFlat *holdAccumGradients, structureC<int> *testCase, dataC* data,
	CNNStructureC *testStruct, size_t begin, double* cost, reducedLayerNodesC* nodesForUpdating, 
	const size_t numBlocks, const size_t numThreads);

void calcGradientParts(CNNStructureThrust& holdAccumGradients, const host_vector<int> testCase, handNumberData& data,
	CNNStructureThrust testStruct, size_t begin, size_t end, double& cost) {
	holdAccumGradients.setToZeros();	//Zero this out because of the += later.
	double tempCost = 0;
	for (size_t tSet = begin; tSet < end; ++tSet) {
		testStruct.updateLayers(data.getInputNodes(tSet));
		tempCost += testStruct.calcCost(data.getInputNodes(tSet), data.getOutputNodes(tSet), false);
		// Add to holdAccumGradients for this test set. 
		testStruct.makeGradPass(holdAccumGradients, data.getOutputNodes(tSet));
	}
//Note, cost has nothing to do with gradient, at this point.
	cost = tempCost / double(end-begin);
	cout << "\nCost in calcGradientParts " << cost;
}

template <class myType>
myType*** getCTripleVector(host_vector<host_vector<host_vector<myType>>>& weights) {
//Return a 3-D array of the weights associated with the vector of vectors of vectors weights.

	myType*** p;
	p = new myType**[weights.size()];

	for (int layer = 0; layer < weights.size(); ++layer) {
		p[layer] = new myType*[weights[layer].size()];
		for (int row = 0; row < weights[layer].size(); ++row) {
				p[layer][row] = weights[layer][row].data();
		}
	}
	return (p);
}

weightsC* getCWeightsFromVectors(host_vector<host_vector<host_vector<double>>>& weights) {
	//Return a 3-D array of the weights associated with the vector of vectors of vectors weights.

	weightsC* tempW = new weightsC;

	size_t numLayers = weights.size();
	tempW->numLayers = numLayers;
	tempW->numRows = new size_t[numLayers];
	tempW->numCols = new size_t[numLayers];
	for (size_t layer = 0; layer < numLayers; ++layer) {
		tempW->numRows[layer] = weights[layer].size();
		tempW->numCols[layer] = weights[layer][0].size();
	}

	size_t totSize = 0.;
	for (int layer = 0; layer < weights.size(); ++layer) {
		for (int row = 0; row < weights[layer].size(); ++row) {
			totSize += weights[layer].size()*weights[layer][row].size();
		}
	}

	tempW->values = new double**[weights.size()];

	for (int layer = 0; layer < weights.size(); ++layer) {
		tempW->values[layer] = new double*[weights[layer].size()];
		for (int row = 0; row < weights[layer].size(); ++row) {
			for (int row = 0; row < weights[layer].size(); ++row) {
				tempW->values[layer][row] = weights[layer][row].data();
			}
		}
	}
	return (tempW);
}
weightsCFlat* getCWeightsFromVectorsFlat(host_vector<host_vector<host_vector<double>>>& weights) {
	//Return a 3-D array of the weights associated with the vector of vectors of vectors weights.

	weightsCFlat* tempW = new weightsCFlat;   

	size_t numLayers = weights.size();
	tempW->numLayers = numLayers;
	tempW->numRows = new size_t[numLayers];
	tempW->numCols = new size_t[numLayers];
	tempW->startLayer = new size_t[numLayers];
	tempW->startLayer[0] = 0;
	for (size_t layer = 0; layer < numLayers; ++layer) {
		tempW->numRows[layer] = weights[layer].size();
		tempW->numCols[layer] = weights[layer][0].size();
		if (layer > 0) {
			tempW->startLayer[layer] = tempW->startLayer[layer - 1] + tempW->numRows[layer - 1] * tempW->numCols[layer - 1];
		}
	}

	size_t totSize = 0.;
	for (int layer = 0; layer < weights.size(); ++layer) {
		for (int row = 0; row < weights[layer].size(); ++row) {
			totSize += weights[layer].size()*weights[layer][row].size();
		}
	}

	tempW->values = new double[totSize];
	tempW->length = totSize;

	for (int layer = 0; layer < weights.size(); ++layer) {
		for (int row = 0; row < weights[layer].size(); ++row) {
			for (int col = 0; col < weights[layer][0].size(); ++col) {
				tempW->values[tempW->startLayer[layer] + row * tempW->numCols[layer] + col] = weights[layer][row][col];
			}
		}
	}
	return (tempW);
}

template <class myType>
myType** getCDoubleVector(host_vector<host_vector<myType>>& layerNodes) {
	//Return a 2-D array of the layer nodes associated with the vector of vectors layerNodes.
	double** p;
	p = new myType*[layerNodes.size()];

	for (int layer = 0; layer < layerNodes.size(); ++layer) {
		p[layer] = layerNodes[layer].data();
	}
	return (p);
}

layerNodesC*  getCLayerNodesFromVectors(host_vector<host_vector<double>>& layerNodes) {
	//Return a 2-D array of the layer nodes associated with the vector of vectors layerNodes.
	double** cLayerNodes = getCDoubleVector<double>(layerNodes);
	layerNodesC* tempLayerNodesC = new layerNodesC;
	tempLayerNodesC->values = cLayerNodes;
	tempLayerNodesC->numLayers = layerNodes.size();
	tempLayerNodesC->numNodes = new int[layerNodes.size()];
	for (size_t layer = 0; layer < layerNodes.size(); ++layer) {
		tempLayerNodesC->numNodes[layer] = layerNodes[layer].size();
	}

	return tempLayerNodesC;
}

layerNodesCFlat*  getCLayerNodesFromVectorsFlat(host_vector<host_vector<double>>& layerNodes) {
	//Return a flattened array of the layer nodes associated with the vector of vectors layerNodes.
//	double** cLayerNodes = getCDoubleVector<double>(layerNodes);

	layerNodesCFlat* tempLayerNodesC = new layerNodesCFlat;
	size_t tot = 0, numLayers;
	numLayers = layerNodes.size();
	for (size_t layer = 0; layer < layerNodes.size(); ++layer) {
		tot += layerNodes[layer].size();
	}

	tempLayerNodesC->startLayer = new size_t[numLayers];
	tempLayerNodesC->startLayer[0] = 0;
	tempLayerNodesC->numLayers = numLayers;
	tempLayerNodesC->numNodes = new int[numLayers];
	for (size_t layer = 0; layer < numLayers; ++layer) {
		tempLayerNodesC->numNodes[layer] = layerNodes[layer].size();
		if (layer > 0) {
			tempLayerNodesC->startLayer[layer] = tempLayerNodesC->startLayer[layer - 1] + tempLayerNodesC->numNodes[layer - 1];
		}
	}

	double* cLayerNodes = new double[tot];
	tempLayerNodesC->length = tot;

	for (size_t layer = 0; layer < layerNodes.size(); ++layer) {
		for (size_t row = 0; row < layerNodes[layer].size(); ++row) {
			cLayerNodes[tempLayerNodesC->startLayer[layer] +row] = layerNodes[layer][row];
		}
	}

	tempLayerNodesC->values = cLayerNodes;

	return tempLayerNodesC;
}

template <class myType>
myType* getCSingleVector(host_vector<myType>& vectorIn) {
	//Return a 2-D array of the layer nodes associated with the vector of vectors layerNodes.

	myType* p;
//	p = new myType[vectorIn.size()];
	p = vectorIn.data();
	return (p);
}
reducedLayerNodesC* setUpLayerNodes(dataC* data, CNNStructureC* testStruct) {
	//Set up layer nodes for each data set. Here I am just creating the memory, which 
	//I expect to duplicate on the device. Note that the input nodes (associated with the images) do not change with 
	//updating. So, I am only making memory for the other layers, including the last one.
	//Calculate the total memory to allocate. Start by summing up the total number of nodes. 

	double*** values;
	values = new double**[data->numSets];
	for (size_t tSet = 0; tSet < data->numSets; ++tSet) {
		values[tSet] = new double*[testStruct->layerNodes.numLayers-1];
		for (size_t layerCount = 0; layerCount < testStruct->layerNodes.numLayers-1; ++layerCount) {
			values[tSet][layerCount] = new double[testStruct->layerNodes.numNodes[layerCount+1]];
		}
	}

	reducedLayerNodesC* nodesForUpdating = new reducedLayerNodesC;
	nodesForUpdating->numSets = data->numSets;

	nodesForUpdating->nodes = values;
	nodesForUpdating->numLayers = testStruct->layerNodes.numLayers-1;
	nodesForUpdating->nodeLengths = new size_t[nodesForUpdating->numLayers];
	for (size_t layerCount = 0; layerCount < nodesForUpdating->numLayers; ++layerCount) {
		nodesForUpdating->nodeLengths[layerCount] = testStruct->layerNodes.numNodes[layerCount+1];
	}

	return (nodesForUpdating);
}

void check(cudaError x) {
	fprintf(stderr, "%s\n", cudaGetErrorString(x));
}

void showMatrix2(int* v1, int width, int height) {
	printf("---------------------\n");
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			printf("%d ", v1[i * width + j]);
		}
		printf("\n");
	}
}
// for cuda error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "\nFatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return 1; \
        } \
    } while (0)



#define SIZE 1024

int main(int argc, char **argv) {

//CUDA test
	int *a, *b, *c;

	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

//	launchCuda(a, b, c, SIZE);

	for (int i = 0; i < 10; ++i) {
		cout << "\nc[" << i << "] = " << c[i];
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
//3D example
//Take the 3D example and turn it into something that takes a large set a vectors, multiplies them by the same
//matrix and reduces them into a final vector. 
//Note that syncThreads() only works for a block, in particular, not across a grid of blocks. That has a big impact on
//the reduction. If you are using a grid of blocks to do a number of layers that exceeds 1024, they won't get synched
//when they need to be. If you needed to reduce a million vectors, you could do this by calling two kernels.
//The first would change the million to a thousand, the second would reduce it to one. I tried using different 
//dimension blocks, (x, y, and z) to exceed the 1024. As far as I can tell, you cannot launch a kernel with 
//blockSize.x*blockSize.y*blockSize.z bigger than 1024. I tried using the y dimension to add to the number of threads
// (+threadidx.y*blockDimx) but, that failed to launch when I tried 2048. So, at this point, I'm going with I cannot
// reduce something having more than 2048 items in only one kernel. 2048 instead of 1024 because I only need half
//as many threads as the dimension to do the reduction. 
//When synching is not an issue, use grids to get as many threads as you want. 
 
	#define BLKXSIZE 1024
	#define BLKYSIZE 1
	#define BLKZSIZE 1

	#define NUMLAYERS 2048
	#define NUMROWS 75
	#define NUMCOLS 75

	dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
	dim3 gridSize(((NUMLAYERS + BLKXSIZE - 1) / BLKXSIZE), ((NUMROWS + BLKYSIZE - 1) / BLKYSIZE), ((NUMCOLS + BLKZSIZE - 1) / BLKZSIZE));
	// overall data set sizes
	const int nx = NUMLAYERS;
	const int ny = NUMROWS;
	const int nz = NUMCOLS;
	// pointers for data set storage via malloc. I want one constant matrix and one set of arrays that will be multiplied
	// on the right. And it lookis like I need a set of arrays to gather up the answer. They will all be flat. 
	double *constMatrix; // storage for result stored on host
	double *d_constMatrix;  // storage for result computed on device
	double *varArray, *d_varArray, *gatherArray, *d_gatherArray,*outArray, *d_outArray;
	// allocate storage for data set
	if ((constMatrix = (double *)malloc((ny*nz) * sizeof(double))) == 0) { fprintf(stderr, "malloc1 Fail \n"); return 1; }
	if ((varArray = (double *)malloc((nx*nz) * sizeof(double))) == 0) { fprintf(stderr, "malloc1 Fail for variable array \n"); return 1; }
	if ((gatherArray = (double *)malloc((nx*ny) * sizeof(double))) == 0) { fprintf(stderr, "malloc1 Fail for gather array \n"); return 1; }
	if ((outArray = (double *)malloc((ny) * sizeof(double))) == 0) { fprintf(stderr, "malloc1 Fail for out array \n"); return 1; }

	// allocate GPU device buffers
	cudaMalloc((void **)&d_constMatrix, (ny*nz) * sizeof(double));
	cudaMalloc((void **)&d_varArray, (nx*nz) * sizeof(double));
	cudaMalloc((void **)&d_gatherArray, (nx*ny) * sizeof(double));
	cudaMalloc((void **)&d_outArray, (ny) * sizeof(double));

	cudaDeviceProp deviceProp;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProp, device);
	std::printf("\nCUDA device capability %d.%d\n", deviceProp.major, deviceProp.minor);

//Fill the matrix. Start with an identity matrix.
	for (size_t i = 0; i < ny; i++) {
		for (size_t j = 0; j < nz; j++) {
			size_t index = i*nz+j;
			constMatrix[index] = .0;
			if (i == j) {
				constMatrix[index] = 1.;
			}
		}
	}
//Fill the varArrays
	for (size_t layer = 0; layer < nx; ++layer) {
		for (size_t j = 0; j < nz; j++) {
			size_t index = layer*nz+ j;
			varArray[index] = double(j);
		}
	}
//You'll be doing += on gather, so initialize it here.

	for (size_t j = 0; j < ny; j++) {
		gatherArray[j] = 0.;
	}

	cudaMemcpy(d_constMatrix, constMatrix, (ny*nz) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_varArray, varArray, (nx*nz) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gatherArray, gatherArray, (nx*ny) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_outArray, outArray, (ny) * sizeof(double), cudaMemcpyHostToDevice);

	cudaCheckErrors("Failed to allocate device buffer");

	cout << "\nblockSize before " << blockSize.x << " " << blockSize.y << " " << blockSize.z;
	cout << "\ngridSize before " <<gridSize.x << " " << gridSize.y << " " << gridSize.z;

	launchCudaMatVecMult(d_constMatrix, d_varArray, d_gatherArray, blockSize, gridSize, nx, ny, nz);
	cout << "\nBack from matrix vec mult";
	cudaCheckErrors("Kernel launch failure");

//The reduction takes half as many threads as layers. I'm going to cut the blockSize here. 
//Not sure if it is worth the effort or risk. Note, reducing it here shows in the next print.
//However, it does not show when I cout blocksize back in the main. 

	blockSize.x = nx / 2;
	blockSize.y = 1;
	blockSize.z = 1;

	gridSize.x = 1;
	gridSize.y = 1;
	gridSize.z = 1;

	launchCudaPVecReduce(d_constMatrix, d_varArray, d_gatherArray, blockSize, gridSize, nx, ny, nz, d_outArray,false);

	cout << "\nblockSize after " << blockSize.x << " " << blockSize.y << " " << blockSize.z;
	cout << "\ngridSize after " << gridSize.x << " " << gridSize.y << " " << gridSize.z;
	cudaCheckErrors("Kernel launch failure");
	// copy output data back to host. I only need the out array.

	cudaMemcpy(outArray, d_outArray, ((ny) * sizeof(double)), cudaMemcpyDeviceToHost);
	cudaCheckErrors("CUDA memcpy failure");

//Do a CPU version of the matVecMult and reduction, to compare with the GPU.
	double** cpuGatherArray = new double*[NUMLAYERS];
	for (size_t iCnt = 0; iCnt < NUMLAYERS; ++iCnt) {
		cpuGatherArray[iCnt] = new double[NUMROWS];
	}

	for (size_t layer = 0; layer < NUMLAYERS; ++layer) {
		for (size_t row = 0; row < NUMROWS; ++row) {
			double temp = 0;
			for (size_t col = 0; col < NUMCOLS; ++col) {
				temp += constMatrix[ row * NUMCOLS + col] * varArray[layer*NUMCOLS + col];
			}
			cpuGatherArray[layer][row] = temp;
//			cout << "\ntemp " << temp;
		}
	}

//Do the reduction. 

	for (size_t s = NUMLAYERS / 2; s > 0; s >>= 1) {
		for (size_t layer = 0; layer < NUMLAYERS / 2; ++layer) {
			if (layer < s) {
				for (size_t row = 0; row < NUMROWS; ++row) {
					cpuGatherArray[layer][row] += cpuGatherArray[layer + s][row];
				}
			}
		}
	}

	cout << "\ncpu reduction ";
	for (size_t row = 0; row < NUMROWS; ++row) {
		cout << cpuGatherArray[0][row]<<" ";
	}

	cout << "\nReduced out\n";
	for (size_t row = 0; row < NUMROWS; ++row) {
		cout << outArray[row] << " ";
	}

//	printf("\nResults check!\n");

	free(constMatrix);
	free(varArray);
	free(gatherArray);
	free(outArray);
	
	cudaFree(d_constMatrix);
	cudaFree(d_varArray);
	cudaFree(d_gatherArray);
	cudaFree(d_outArray);

	cudaCheckErrors("cudaFree fail");
	cudaDeviceReset();

	//Set up the training data.

	double gradientCutDown = 200.;
	size_t lapCounter = 0, numBetweenPrints = 9, numSinceLastPrint = 0;
	// Set up a test case for the structure
	host_vector<int> testCase;
	vector<int> junk;	//Note this indicates that you might not need thrust vec
	string fileNameLabels = "../project1/data/t10k-labels.idx1-ubyte";
	string fileNameImages = "../project1/data/t10k-images.idx3-ubyte";

	handNumberData data1(fileNameImages, fileNameLabels);
	data1.displayImage(5);

	//The following determines the size of the network. The input dimension determines the columns of the first
	//matrix. The number of rows of the first matrix is determined by the number in the next layer. The following
	//makes a four layer system(two hidden layers).

	testCase.push_back((int)data1.getInputDimension());
	testCase.push_back(16);
	testCase.push_back(16);
	testCase.push_back((int)data1.getOutputDimension());

	/*
		dataSet data1(80), data2(40);

		testCase.push_back((int)data1.getInputDimension());
		testCase.push_back(3);
		testCase.push_back(3);
		testCase.push_back((int)data1.getOutputDimension());

	*/
	cout << "\ndata1.getOutputDimension() " << data1.getOutputDimension();
	cout << "\ndata1.getInputDimension() " << data1.getInputDimension();

	//	CNNStructureThrust testStruct(testCase, .5, 1.);holdAccumGradients
	string inFile = "../project1/states/10kweightsFile9.txt";
	//	string inFile = "./states/testWeights.txt";

	string outFile = "../project1/states/10kweightsFile10.txt";
	size_t numTrainingLoops = 30;

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
	const size_t numThreads = 1024;

	//You should make a function for the following so that you are not doing things twice.
	CNNStructureThrust testStruct(inFile);
	//Create an indepentent CNNStructure for holding the gradients.
//	CNNStructureThrust holdAccumGradients(testCase);
	vector<CNNStructureThrust> holdAccumGradients(numThreads, CNNStructureThrust(testCase));
	host_vector<double> costHistory;
	costHistory.reserve(numTrainingLoops);
	//Get the starting cost.
	double tempCost = 0.;
	for (size_t tSet = 0; tSet < data1.getNumSets(); ++tSet) {
		tempCost += testStruct.calcCost(data1.getInputNodes(tSet), data1.getOutputNodes(tSet)) / double(data1.getNumSets());
	}
	cout << "\nStarting cost " << tempCost;
	costHistory.push_back(tempCost);

	//Make the testStructure copy.

	weightsC* testWeightsC = getCWeightsFromVectors(testStruct.getWeights());
	layerNodesC* testLayerNodesC = getCLayerNodesFromVectors(testStruct.getLayerNodes());

	float totCount = 0, failCount = 0;
#ifdef _DEBUG
	//Check testLayerNodesC against testStruct layerNodes

	for (size_t layer = 0; layer < testStruct.getNumWeightsMatrices(); ++layer) {
		for (size_t row = 0; row < testStruct.getLayerNodes()[layer].size(); ++row) {

			++totCount;
			if (testLayerNodesC->values[layer][row] != testStruct.getLayerNodes()[layer][row]) {
				++failCount;
				cout << "\ntestLayerNodesC[layer][row] " << testLayerNodesC->values[layer][row] << " and "
					<< testStruct.getLayerNodes()[layer][row];
			}
		}
	}
	cout << "\nChecking layernodes total checked " << totCount << " and error count " << failCount;
#endif

//Now make flattened versions. Note, that I plan to start with flattened arrays of the actual values,
//but still store them in structures (not making the entire structure flat).

	weightsCFlat* testWeightsCFlat = getCWeightsFromVectorsFlat(testStruct.getWeights());
	layerNodesCFlat* testLayerNodesCFlat = getCLayerNodesFromVectorsFlat(testStruct.getLayerNodes());
#ifdef _DEBUG
	//Check testLayerNodesCFlat against testStruct.getLayerNodes()
	totCount = 0, failCount = 0;
	for (size_t layer = 0; layer < testStruct.getNumWeightsMatrices(); ++layer) {
		for (size_t row = 0; row < testStruct.getLayerNodes()[layer].size(); ++row) {

			++totCount;
			if (testLayerNodesCFlat->values[testLayerNodesCFlat->startLayer[layer]+row]
				!= testStruct.getLayerNodes()[layer][row]) {
				++failCount;
				cout << "\nLayer and row " <<layer<<" "<<row<<" Flat = "<<
					testLayerNodesCFlat->values[testLayerNodesCFlat->startLayer[layer] + row] << " and reg = "
					<< testStruct.getLayerNodes()[layer][row];
			}
		}
	}
	cout << "\nChecking layernodesFlat total checked " << totCount << " and error count " << failCount;
	

	//Check testWeightsCFlat against cWeights.
	cout << "\nChecking flattened weights ";
	size_t numErrors = 0;
	for (size_t layer = 0; layer < testWeightsCFlat->numLayers; ++layer) {
		for (size_t row = 0; row < testWeightsCFlat->numRows[layer]; ++row) {
			for (size_t col = 0; col < testWeightsCFlat->numCols[layer]; ++col) {
				double c1, c2;
				c1 = testWeightsC->values[layer][row][col];
				c2 = testWeightsCFlat->values[testWeightsCFlat->startLayer[layer] + testWeightsCFlat->numCols[layer] * row + col];
				if (c1 != c2) {
					cout << "\nlayer " << layer;
					cout << "\nrow " << row;
					cout << "\n " << testWeightsC->values[layer][row][col] << " = " <<
						testWeightsCFlat->values[testWeightsCFlat->startLayer[layer] + testWeightsCFlat->numCols[layer] * row + col];
					++numErrors;
				}
			}    
		}
	}
	if (numErrors != 0) {
		cout << "\n Found " << numErrors << " differences";
	}
	else {
		cout << "\nNo errors found for cWeightsFlat ";
	}
#endif

	weightsC** testWeightsCTemp = new weightsC*[numThreads];	//Making an array of ptrs.
	layerNodesC** testLayerNodesCTemp = new layerNodesC*[numThreads];

	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {

		testWeightsCTemp[iCnt] = getCWeightsFromVectors(holdAccumGradients[iCnt].getWeights());
		testLayerNodesCTemp[iCnt] = getCLayerNodesFromVectors(holdAccumGradients[0].getLayerNodes());
	}
	weightsCFlat** testWeightsCTempFlat = new weightsCFlat*[numThreads];	//Making an array of ptrs.
	layerNodesCFlat** testLayerNodesCTempFlat = new layerNodesCFlat*[numThreads];

	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {

		testWeightsCTempFlat[iCnt] = getCWeightsFromVectorsFlat(holdAccumGradients[iCnt].getWeights());
		testLayerNodesCTempFlat[iCnt] = getCLayerNodesFromVectorsFlat(holdAccumGradients[0].getLayerNodes());
	}

	CNNStructureC testStructC;
	testStructC.weights = *testWeightsC;
	testStructC.layerNodes = *testLayerNodesC;

	CNNStructureCFlat testStructCFlat;
	testStructCFlat.weights = *testWeightsCFlat;
	testStructCFlat.layerNodes = *testLayerNodesCFlat;

	CNNStructureC holdAccumGradientsC[numThreads];
	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {
		holdAccumGradientsC[iCnt].weights = *testWeightsCTemp[iCnt];
		holdAccumGradientsC[iCnt].layerNodes = *testLayerNodesCTemp[iCnt];
	}

	CNNStructureCFlat holdAccumGradientsCFlat[numThreads];
	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {
		holdAccumGradientsCFlat[iCnt].weights = *testWeightsCTempFlat[iCnt];
		holdAccumGradientsCFlat[iCnt].layerNodes = *testLayerNodesCTempFlat[iCnt];
	}

	int* tempCaseC = getCSingleVector<int>(testCase);
	structureC<int> testCaseC;
	testCaseC.structure = tempCaseC;
	testCaseC.numElements = testCase.size();
#ifdef _DEBUG
	for (int layer = 0; layer < testCaseC.numElements; ++layer) {
		cout << "\ntestCase C and V " << testCaseC.structure[layer] << " " << testCase[layer];
	}
#endif
//Copy data to a dataC. I think that each of the inputs and outputs for each set is already flat. Just like with the 
//structures, I don't know if further flattening is necessary. 
	dataC  data1C;
	data1C.numInputs = data1.getInputDimension();	
	cout << "\ntestLayerNodesC->numLayers " << testLayerNodesC->numLayers;
	data1C.numOutputs = data1.getOutputDimension();		
	cout << "\n Should be 785 " << data1C.numInputs << " and 11 " << data1C.numOutputs;
	data1C.numSets = data1.getNumSets();
	data1C.labels = new int[data1.getNumSets()];

	data1C.inputNodes = new double*[data1.getNumSets()];
	data1C.outputNodes = new double*[data1.getNumSets()];
	for (size_t sets = 0; sets < data1.getNumSets(); ++sets) {

		data1C.inputNodes[sets] = data1.getInputNodes(sets).data();
		data1C.outputNodes[sets] = data1.getOutputNodes(sets).data();

		data1C.labels[sets] = data1.getLabel(sets);
	}
//Make the memory for the reduced nodes.
	reducedLayerNodesC* nodesForUpdating;
	nodesForUpdating = setUpLayerNodes(&data1C, &testStructC);

	cout << "\n should be 10000 " << nodesForUpdating->numSets;
	cout << "\n Should be 3 " << nodesForUpdating->numLayers;
	for (size_t layerCount = 0; layerCount < nodesForUpdating->numLayers; ++layerCount) {
		cout << "\n layerCount " <<layerCount<<" "<< nodesForUpdating->nodeLengths[layerCount];
	}
#ifdef _DEBUG
	//Make sure the output nodes match the labels.
	for (int sets = 0; sets < 20; ++sets) {
		cout << "\n label " << data1C.labels[sets] <<" "<<data1.getLabel(sets) <<"\n";
		for (int iCnt = 0; iCnt < data1C.numOutputs; ++iCnt) {
			cout << data1C.outputNodes[sets][iCnt];
		}
		cout << "\n";
		for (int iCnt = 0; iCnt < data1C.numOutputs; ++iCnt) {
			cout << data1.getOutputNodes(sets)[iCnt];
		}
	}
	cout << "\n*********************** End test***************************";
#endif


//Start training
	size_t begin = 0;
	size_t end = data1.getNumSets();	
	const size_t numBlocks = end / numThreads;
	cout << "\nNumber of blocks and threads " << numBlocks << " " << numThreads;

	double costFromGradCalc;
	double costFromGradCalcC;
	auto beginLoopTime = chrono::high_resolution_clock::now();

//Let's start by putting the weights for holdAccumGradientsCFlat on the device.
	double ** d_holdAccumGradientsCFlatWeights = new double*[numThreads];

	for (size_t set = 0; set < numThreads; ++set) {
		cudaMalloc((void **)&d_holdAccumGradientsCFlatWeights[set],holdAccumGradientsCFlat[set].weights.length * sizeof(double));
//The following is what sets the values on the device. However, in this case, there are no values. They will be set to zero
//on the device. It may be possible to skip the Memcpy. 
		cudaMemcpy(d_holdAccumGradientsCFlatWeights[set],
			holdAccumGradientsCFlat[set].weights.values, holdAccumGradientsCFlat[set].weights.length * sizeof(double),
			cudaMemcpyHostToDevice);
	}
	for (size_t trainLoops = 0; trainLoops < numTrainingLoops; ++trainLoops) {
		/*
GPU discussion. I expect to see significant savings when I put all the data on the device before entering
this training loop and only occasionally getting just the reduced cost number out after each iteration
through the entire training set. Until that, my plan is to add kernels that do their part in the pipeline
returning results back to the CPU to continue the pipeline. The idea is that I'll be able to check that each
kernel has performed the same as the previous CPU version. I don't expect time savings from that. Just an
incremental method for getting to the savings. The first kernel will just set the holdAccumGradientCFlat arrays
to zero.
*/

//Note that holdAccumGradients and testStruct use different memory. However, holdAccumGradients and holdAccumGradientsC
//share the same memory. As does testStruct and testStruct C. In the following, testStruct is not supposed to change.
//holdAccum... does. Each should calculate the same change. But, to compare, you need to observe them as 
//they are being created, since at any time they are both the same, i.e., watch the numbers going into holdAcc.. inside
//the respective functions. 
//		calcGradientParts(holdAccumGradients, testCase, data1, testStruct, begin, end, costFromGradCalc);

//		calcGradientPartsC(holdAccumGradientsC, &testCaseC, &data1C, &testStructC, begin, &costFromGradCalcC, 
//			nodesForUpdating, numBlocks, numThreads);

		calcGradientPartsCUDA(holdAccumGradientsCFlat, &testCaseC, &data1C, &testStructC, begin, &costFromGradCalcC, 
			nodesForUpdating, numBlocks, numThreads);

// Normalize by dividing by the number of entries. You may want to do other things to reduce it as well.
//		holdAccumGradients[0].divideScaler(double(-gradientCutDown * (data1.getNumSets())));
		for (size_t i = 0; i < holdAccumGradientsCFlat->weights.length; ++i) {
			holdAccumGradientsCFlat[0].weights.values[i] /= (double(-gradientCutDown * (data1.getNumSets())));
		}

//Since holdAccumGradients and holdAccumGradientsC both point to the same memory. It does not matter whether you 
//update holdAccumGradients or holdAccumGradientsC. Either way, you have updated holdAccumGradients below. So,
//you can use vectors again.

// Modify the weights and biases.

		size_t indexFlattened = 0;
		for (size_t layer = 0; layer < testStruct.getNumWeightsMatrices(); ++layer) {//Fix this kluge when testStruct is flattened.
			for (size_t row = 0; row < testStruct.getNumWeightsRows(layer); ++row) {
				for (size_t col = 0; col < testStruct.getNumWeightsCols(layer); ++col) {
					holdAccumGradients[0].setWeights(layer, row, col, 
						holdAccumGradientsCFlat[0].weights.values[indexFlattened]);
					++indexFlattened;

				}
			}
		}
		testStruct += holdAccumGradients[0];	 

		costHistory.push_back(costFromGradCalcC);

		++lapCounter;
		//		if (lapCounter > 5)gradientCutDown = 20.;
		//		if (lapCounter > 200)gradientCutDown = 10.;

		if (numSinceLastPrint > numBetweenPrints) {

			cout << "\ncost " << costFromGradCalcC;

			numSinceLastPrint = 0;
			auto endLoopTime = chrono::high_resolution_clock::now();
			cout << " [" << lapCounter << "]  Time " <<
				chrono::duration<double>(endLoopTime - beginLoopTime).count();
			beginLoopTime = chrono::high_resolution_clock::now();

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

	//for (size_t tSet = data1.getNumSets(); tSet < data1.getNumSets(); ++tSet) {
	for (size_t tSet = 0; tSet < 200; ++tSet) {
		statTestTotal[data1.getLabel(tSet)]++;
		testStruct.updateLayers(data1.getInputNodes(tSet));

		testStruct.displayLayerNodes(testStruct.getNumWeightsMatrices());

		double max = -DBL_MAX;
		size_t indexToKeep = 0;
		for (size_t iCnt = 0; iCnt < data1.getOutputDimension() - 1; ++iCnt) {
			//Find the largest
			if (testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[iCnt] > max) {
				max = testStruct.getLayerNodes(testStruct.getNumWeightsMatrices())[iCnt];
				indexToKeep = iCnt;
			}
		}
		cout << "Chosen index " << indexToKeep << " label " << data1.getLabel(tSet) << "\n\n";
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
		cout << "\ncount " << iCnt << " total " << statTestTotal[iCnt] << " and missed " << statTestMissed[iCnt];
	}

	return 0;
}