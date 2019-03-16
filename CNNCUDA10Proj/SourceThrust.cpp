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
	CNNStructureC *testStruct, size_t begin, double* cost, reducedLayerNodesC* nodesForUpdating, const size_t numBlocks, const size_t numThreads);


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
//	cudaMemcpy(gatherArray, d_gatherArray, ((ny*nx) * sizeof(double)), cudaMemcpyDeviceToHost); //Just to check intermediate
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

	// Compare cpuGatherArray with gatherArray without calling reduction.
	//MatrixVec multiplication. Note, you can only check this when gatherArray has not been reduced. 
	/*

	size_t numChecked = 0, numErrors = 0;
	for (size_t layer = 0; layer < NUMLAYERS; ++layer) {
		for (size_t row = 0; row < NUMROWS; ++row) {
			numChecked++;
			if (cpuGatherArray[layer][row] != gatherArray[layer*NUMROWS + row]) {
				numErrors++;
				cout << "\nMismatch between cpu and gpu " << cpuGatherArray[layer][row] << " " << gatherArray[layer*NUMROWS + row];
			}
		}
	}
	cout << "\nChecked " << numChecked << " values with " << numErrors << " errors ";
*/

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

//Compare reductions
/*
	size_t numChecked = 0, numErrors = 0;
	for (size_t layer = 0; layer < NUMLAYERS; ++layer) {
//		cout << "\nlayer " << layer;
		for (size_t row = 0; row < NUMROWS; ++row) {
//			cout << "\nrow " << row;
//			cout << "\ncpu and gpu " << cpuGatherArray[layer][row] << " " << gatherArray[layer*NUMROWS + row];
			numChecked++;
			if (cpuGatherArray[layer][row] != gatherArray[layer*NUMROWS + row]) {
				numErrors++;
			
				cout << "\nMismatch between cpu and gpu " << cpuGatherArray[layer][row] << " " <<
					gatherArray[layer*NUMROWS + row] << " layer = " << layer << " row = " << row;
			}
		}
	}
	cout << "\nChecked " << numChecked << " values with " << numErrors << " errors ";
*/
/*	// and check for accuracy
	for(size_t layer = 0; layer<nx;++layer){
//		cout << "\n layer " << layer;
		for (unsigned row = 0; row < ny; row++) {
//			cout << "\nRow " << j << " ";
			size_t index = layer * ny + row;
//			cout << gatherArray[index] << " ";

			if (gatherArray[index] != double(row)) {
				printf("\nMismatch gatherArray[index] %f layer= %d, row= %d  ",gatherArray[index], layer, row);
//				return 1;
			}
		}
	}
*/	
//I'm going to make a CPU version of the reduction here. Threadx (layer) will come from blockSize.x
//And thready (row) will come from blockSize.y. The total number of layers (to be summed up) is gridSize.x * blockSize.x
/*
	for (size_t layer = blockSize.x*gridSize.x/2; layer > 0; layer >>= 1) {
		for (size_t subLayer = 0; subLayer < layer; ++subLayer) {
			for (size_t row = 0; row < NUMROWS; ++row) {
				size_t index0 = subLayer * NUMROWS + row;
				size_t index1 =  (subLayer+layer)*NUMROWS + row;
				if (index1 < NUMLAYERS*NUMROWS) {
					gatherArray[index0] += gatherArray[index1];

				}
			}
		}
	}
	*/
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
	exit(0);

/*
//Pitch method
	cout << "\nFirst pitch example" << "\n";
	int width = 4;
	int height = 4;

	int* d_tab;
	int* h_tab;

	int realSize = width * height * sizeof(int);

	size_t pitch;
	check(cudaMallocPitch(&d_tab, &pitch, width * sizeof(int), height));
	h_tab = (int*)malloc(realSize);
	check(cudaMemset(d_tab, 0, realSize));

	launchCudaPitch(width, height, pitch, d_tab);

	check(cudaMemcpy2D(h_tab, width * sizeof(int), d_tab, pitch, width * sizeof(int), height, cudaMemcpyDeviceToHost));

	showMatrix2(h_tab, width, height);
	printf("\nPitch size: %ld \n", pitch);

	exit(0);

	//End CUDA test
*/
//Do some memory experiments
/*
	size_t memSize = 100000000;
	double* checkMem = new double[memSize];
	for (size_t iCnt = 0; iCnt < memSize; ++iCnt) {
		checkMem[iCnt] = 1.;
	}
	double* d_checkMem;
	cudaMalloc((void**)&d_checkMem, memSize * sizeof(double));
	cudaMemcpy(d_checkMem, checkMem, memSize * sizeof(double), cudaMemcpyHostToDevice);

	if (checkMem) {
		cout << "\nAllocated memory for " << memSize << " doubles";

	}
	else
	{
		cout << "\ncould not allocate memory of " << memSize << " doubles";
	}
	int junk;
	cin >> junk;
	exit(0);
*/
// Let's try a class example. 
/*
	size_t numRows(5), numCols(7);
	size_t numLoops = 10000;
	cout << "\n Making myMatrix1 ";
	SimpleMatrix** myMatrix1;
	SimpleMatrix** myMatrix2;

	cudaMallocManaged(&myMatrix1, numLoops * sizeof(double));
	cudaMallocManaged(&myMatrix2, numLoops * sizeof(double));

	cout << "\nsize " << sizeof(myMatrix1);
	cout << "\n Made first layer  memory ";
	for (size_t loopCount = 0; loopCount < numLoops; ++loopCount) {
//		cout << "\nloopCount should make it to " << loopCount << " " << numLoops;

		myMatrix1[loopCount] = new SimpleMatrix(numRows, numCols, .00001*double(loopCount));

		if (myMatrix1[loopCount] == nullptr) {
			cout << "\nMatrix1 Error at "<<loopCount;
			exit(0);
		}
		myMatrix2[loopCount] = new SimpleMatrix(numRows, numCols, .3);
		if (myMatrix2[loopCount] == nullptr) {
			cout << "\nMatrix2 Error at " << loopCount;
			exit(0);
		}
	}

	cout << "\nMatrix1[1] result " << "\n";
	for (size_t row = 0; row < numRows; ++row) {
		cout << "\n row = " << row;
		for (size_t col = 0; col < numCols; ++col) {
			cout << " " << myMatrix1[1]->getMatrixElem(row, col);
		}
	}
	cout << "\nMatrix2[1] result " << "\n";
	for (size_t row = 0; row < numRows; ++row) {
		cout << "\n row = " << row;
		for (size_t col = 0; col < numCols; ++col) {
			cout << " " << myMatrix2[1]->getMatrixElem(row, col);
		}
	}

	launchCudaMatrix(numLoops, myMatrix1, myMatrix2);

	cout << "\nMatrix1[0] result after CUDA" << "\n";
	for (size_t row = 0; row < numRows; ++row) {
		cout << "\n row = " << row;
		for (size_t col = 0; col < numCols; ++col) {
			cout << " " << myMatrix1[1]->getMatrixElem(row, col);
		}
	}

	cudaFree(myMatrix1);
	cudaFree(myMatrix2);

	exit(0);
*/
// End of class example
	//Set up the training data.

	double gradientCutDown = 200.;
	size_t lapCounter = 0, numBetweenPrints = 9, numSinceLastPrint = 0;
	// Set up a test case for the structure
	host_vector<int> testCase;
	vector<int> junk;	//Note this indicates that you do not need thrust vec
	string fileNameLabels = "../project1/data/t10k-labels.idx1-ubyte";
	string fileNameImages = "../project1/data/t10k-images.idx3-ubyte";

	handNumberData data1(fileNameImages, fileNameLabels);
	data1.displayImage(5);
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

	double*** cWeights = getCTripleVector<double>(testStruct.getWeights());
	double*** cWeightsTemp[numThreads];
	double** cLayerNodesTemp[numThreads];
	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {
		cWeightsTemp[iCnt] = getCTripleVector<double>(holdAccumGradients[iCnt].getWeights());
		cLayerNodesTemp[iCnt] = getCDoubleVector<double>(holdAccumGradients[0].getLayerNodes());
	}
//Make the testStructure copy.
	weightsC testWeightsC;

	testWeightsC.values = cWeights;
	testWeightsC.numLayers = testStruct.getNumWeightsMatrices();
	testWeightsC.numRows = new int[testWeightsC.numLayers];
	testWeightsC.numCols = new int[testWeightsC.numLayers];

	for (size_t layer = 0; layer < testWeightsC.numLayers; layer++) {
		testWeightsC.numRows[layer] = testStruct.getNumWeightsRows(layer);
		testWeightsC.numCols[layer] = testStruct.getNumWeightsCols(layer);
	}
	double** cLayerNodes = getCDoubleVector<double>(testStruct.getLayerNodes());
	layerNodesC testLayerNodesC;
	testLayerNodesC.values = cLayerNodes;
	testLayerNodesC.numLayers = testCase.size();
	testLayerNodesC.numNodes = new int[testCase.size()];

// Make holdAccumGradients copy(ies). 
	weightsC testWeightsCTemp[numThreads];
	layerNodesC testLayerNodesCTemp[numThreads];
	for (size_t iCnt=0; iCnt < numThreads; ++iCnt) {
		testWeightsCTemp[iCnt].values = cWeightsTemp[iCnt];
		testWeightsCTemp[iCnt].numLayers = holdAccumGradients[iCnt].getNumWeightsMatrices();
		testWeightsCTemp[iCnt].numRows = new int[testWeightsCTemp[iCnt].numLayers];
		testWeightsCTemp[iCnt].numCols = new int[testWeightsCTemp[iCnt].numLayers];

		for (size_t layer = 0; layer < testWeightsCTemp[iCnt].numLayers; layer++) {
			testWeightsCTemp[iCnt].numRows[layer] = holdAccumGradients[iCnt].getNumWeightsRows(layer);
			testWeightsCTemp[iCnt].numCols[layer] = holdAccumGradients[iCnt].getNumWeightsCols(layer);
		}
		testLayerNodesCTemp[iCnt].values = cLayerNodesTemp[iCnt];
		testLayerNodesCTemp[iCnt].numLayers = testCase.size();
		testLayerNodesCTemp[iCnt].numNodes = new int[testCase.size()];

		for (size_t layer = 0; layer < testCase.size(); ++layer) {
			testLayerNodesCTemp[iCnt].numNodes[layer] = testCase[layer];
		}
	}

	for (size_t layer = 0; layer < testCase.size(); ++layer) {
		testLayerNodesC.numNodes[layer] = testCase[layer];
	}

	CNNStructureC testStructC;
	testStructC.weights = testWeightsC;
	testStructC.layerNodes = testLayerNodesC;

	CNNStructureC holdAccumGradientsC[numThreads];
	for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {
		holdAccumGradientsC[iCnt].weights = testWeightsCTemp[iCnt];
		holdAccumGradientsC[iCnt].layerNodes = testLayerNodesCTemp[iCnt];
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
//Copy data to a dataC.
	dataC  data1C;
	data1C.numInputs = data1.getInputDimension();	
	cout << "\ntestLayerNodesC.numLayers " << testLayerNodesC.numLayers;
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
//Make the memory for the educed nodes.
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
	for (size_t trainLoops = 0; trainLoops < numTrainingLoops; ++trainLoops) {

//end = 10; //Cutting this way back for testing.
//Note that holdAccumGradients and testStruct use different memory. However, holdAccumGradients and holdAccumGradientsC
//share the same memory. As does testStruct and testStruct C. In the following, testStruct is not supposed to change.
//holdAccum... does. Each should calculate the same change. But, to compare, you need to observe them as 
//they are being created, since at any time they are both the same, i.e., watch the numbers going into holdAcc.. inside
//the respective functions. 
//		calcGradientParts(holdAccumGradients, testCase, data1, testStruct, begin, end, costFromGradCalc);

		calcGradientPartsC(holdAccumGradientsC, &testCaseC, &data1C, &testStructC, begin, &costFromGradCalcC, 
			nodesForUpdating, numBlocks, numThreads);

		// Normalize by dividing by the number of entries. You may want to do other things to reduce it as well.
		holdAccumGradients[0].divideScaler(double(-gradientCutDown * (data1.getNumSets())));

//Since holdAccumGradients and holdAccumGradientsC both point to the same memory. It does not matter whether you 
//update holdAccumGradients or holdAccumGradientsC. Either way, you have updated holdAccumGradients below. So,
//you can use vectors again.

// Modify the weights and biases.
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