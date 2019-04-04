/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "CStructs.h"

extern "C" void launchCuda(int* a, int* b, int*c, int n);

extern "C" void launchCudaMatrix(size_t numLoops, SimpleMatrix** myMatrix1, SimpleMatrix** myMatrix2);
//extern "C" void launchCudaMatrix(size_t numLoops, int** myMatrix1, int** myMatrix2);

extern "C" void launchCudaPitch(int width, int height, size_t pitch, int* d_tab);

extern "C" void launchCudaMatVecMult(double *d_constMatrix, double* d_varArray,double* d_gatherArray,
	dim3 blockSize, dim3 gridSize, int nx, int ny, int nz);

extern "C" void launchCudaPVecReduce(double *d_constMatrix, double* d_varArray, double* d_gatherArray,
	dim3 blockSize, dim3 gridSize, int nx, int ny, int nz, double*outArray, bool shared = true);

// device function to set the 3D volume
__global__ void pMatVecMult(double *cM, double *varA, double *gatherA, int nx, int ny, int nz)
{
	unsigned layer = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned row = blockIdx.y*blockDim.y + threadIdx.y;
	double tempSum = 0.;
	if ((layer < nx) &&(row < ny)) {
		for (size_t col = 0; col < nz; ++col) {
			tempSum += cM[row*nz + col] * varA[layer*nz + col];

		}
		gatherA[layer*ny+ row] = tempSum;
	}
}

__global__ void pVecReducedGlobal(double *gatherA, int nx, int ny, int n, double* out)
{
	size_t t_id = ny * (threadIdx.x + blockDim.x*blockIdx.x );

	for(size_t s = nx / 2; s > 0; s >>= 1) {
		if (t_id < s*ny) {
			for (size_t row = 0; row < ny; ++row) {
				gatherA[t_id+row] += gatherA[t_id + s*ny+row];
			}
		}
		__syncthreads();
	}

	if (t_id == 0) {
		for (size_t row = 0; row < ny; ++row) {
			out[row] = gatherA[t_id+row];
		}
	}
}

__global__ void pVecReduceShared(double *gatherA, int nx, int ny, int n, double* out)

{
	extern double __shared__ sdata[];
	size_t t_id = ny * (threadIdx.x + blockDim.x*blockIdx.x);
	for (size_t row = 0; row < ny; ++row) {
		sdata[t_id+row] = gatherA[t_id+row];
		sdata[t_id + nx*ny / 2+row] = gatherA[t_id + nx*ny / 2+row];
	}
	__syncthreads();

	for (size_t s = nx / 2; s > 0; s >>= 1) {
		if (t_id < s*ny) {
			for (size_t row = 0; row < ny; ++row) {
				sdata[t_id + row] += sdata[t_id + s * ny + row];
			}
		}
		__syncthreads();
	}

	if (t_id == 0) {
		for (size_t row = 0; row < ny; ++row) {
			out[row] = sdata[t_id + row];
		}
	}
}

void launchCudaMatVecMult(double *d_constMatrix, double* d_varArray, double* d_gatherArray,
	dim3 blockSize, dim3 gridSize, int nx, int ny, int nz) {

	pMatVecMult << <gridSize, blockSize >> > (d_constMatrix, d_varArray, d_gatherArray, nx, ny, nz);
	cudaDeviceSynchronize();
}

void launchCudaPVecReduce(double *d_constMatrix, double* d_varArray, double* d_gatherArray,
	dim3 blockSize, dim3 gridSize, int nx, int ny, int nz, double*outArray,bool shared) {

	if (shared) {
		pVecReduceShared << <gridSize, blockSize, nx*ny*sizeof(double) >> > (d_gatherArray, nx, ny, nz, outArray);
	}
	else
	{
		pVecReducedGlobal << <gridSize, blockSize >> > (d_gatherArray, nx, ny, nz, outArray);
	}
	cudaDeviceSynchronize();
}

__global__ void pitchEx2D_1(int* tab, int width, int height, size_t pitch) {

	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;

	if (row < width && col < height) {
		*(((int *)(((char *)tab) + (row * pitch))) + col) = 9;
	}
}

void launchCudaPitch(int width, int height, size_t pitch, int* d_tab) {
	dim3 grid(width, height);
	dim3 block(width, height);
	pitchEx2D_1 << <grid, block >> > (d_tab, width, height, pitch);
	cudaDeviceSynchronize();
}

//Sample CUDA function
__global__ void vectorAdd(int* a, int* b, int*c, int n) {
	int i = threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}

void launchCuda(int* a, int* b, int*c, int n) {
	vectorAdd <<< 1, n >>> (a, b, c, n);
	cudaDeviceSynchronize();
}

__global__ void simpleMatrixAdd(size_t numLoops, SimpleMatrix** myMatrix1, SimpleMatrix** myMatrix2,size_t numRows, size_t numCols) {

//	size_t numRows = myMatrix1[0]->numRows;	//Not sure if this was important, since you were not setting numRows or numCols
//	size_t numCols = myMatrix1[0]->numCols;
	int loops = threadIdx.x;
	myMatrix1[1]->matrix[2] = 200.;
	if (loops < numLoops) {
		for (size_t row = 0; row < numRows; ++row) {
			for (size_t col = 0; col <numCols; ++col) {
				myMatrix1[loops]->matrix[row*numCols + col] = myMatrix1[loops]->matrix[row*numCols + col] +
					myMatrix2[loops]->matrix[row*numCols + col];

			}
		}
	}
	myMatrix1[1]->matrix[3] = 200.;
}

void launchCudaMatrix(size_t numLoops, SimpleMatrix** myMatrix1, SimpleMatrix** myMatrix2) {
	myMatrix1[1]->matrix[0] = 100.;
	size_t numIters = 256;
	size_t blocks = numLoops / numIters;
	printf("\nCalling kernel with numLoops = %d",numLoops);
	simpleMatrixAdd << < blocks , numIters >> > (numLoops, myMatrix1, myMatrix2, myMatrix1[0]->numRows, myMatrix1[0]->numCols);
	cudaDeviceSynchronize();
	myMatrix1[1]->matrix[1] = 100.;

}

void matVecMultC(double** weights, double* inVector, int numRows, int numCols, double* p1) {

	for (int rowCount = 0; rowCount < numRows; ++rowCount) {
		double tempSum = 0;
		for (int colCount = 0; colCount < numCols; ++colCount) {
			tempSum += weights[rowCount][colCount] * inVector[colCount];
		}
		p1[rowCount] = tempSum;
	}
}


void plusEqualsStruct(CNNStructureC *holdAccumGradients1, CNNStructureC *holdAccumGradients2) {
	for (size_t numLayer = 0; numLayer < holdAccumGradients1->weights.numLayers; ++numLayer) {
		for (size_t numRow = 0; numRow < holdAccumGradients1->weights.numRows[numLayer]; ++numRow) {
			for (size_t numCol = 0; numCol < holdAccumGradients1->weights.numCols[numLayer]; ++numCol) {
				holdAccumGradients1->weights.values[numLayer][numRow][numCol] +=
					holdAccumGradients2->weights.values[numLayer][numRow][numCol];
			}
		}
	}
};
void setWeightsToZeros(CNNStructureC *input) {
	//Assumes input already has memory created.
	for (int layer = 0; layer < input->weights.numLayers - 1; ++layer) {
		for (int row = 0; row < input->weights.numRows[layer]; ++row) {
			for (int col = 0; col < input->weights.numCols[layer]; ++col) {
				input->weights.values[layer][row][col] = 0;
			}
		}
	}
};
void plusEqualsStruct(CNNStructureCFlat *holdAccumGradients1, CNNStructureCFlat *holdAccumGradients2) {
	size_t indexFlatten;
	for (size_t numLayer = 0; numLayer < holdAccumGradients1->weights.numLayers; ++numLayer) {
		for (size_t numRow = 0; numRow < holdAccumGradients1->weights.numRows[numLayer]; ++numRow) {
			for (size_t numCol = 0; numCol < holdAccumGradients1->weights.numCols[numLayer]; ++numCol) {
				indexFlatten = holdAccumGradients1->weights.startLayer[numLayer] +
					numRow * holdAccumGradients1->weights.numCols[numLayer] + numCol;
				holdAccumGradients1->weights.values[indexFlatten] +=
					holdAccumGradients2->weights.values[indexFlatten];
			}
		}
	}
};


void setWeightsToZeros(CNNStructureCFlat *input) {
	//Assumes input already has memory created.
	for (int layer = 0; layer < input->weights.numLayers - 1; ++layer) {
		for (int row = 0; row < input->weights.numRows[layer]; ++row) {
			for (int col = 0; col < input->weights.numCols[layer]; ++col) {
				input->weights.values[input->weights.startLayer[layer] + row * input->weights.numCols[layer] + col] = 0;
			}
		}
	}
};

void structMatVecMult(CNNStructureC* inStruct, reducedLayerNodesC* nodesForUpdating, double* inVector, int numLayer, int whichCase) {
	int numCols = inStruct->weights.numCols[numLayer];
	int numRows = inStruct->weights.numRows[numLayer];
	for (int rowCount = 0; rowCount < numRows; ++rowCount) {
		double tempSum = 0;
		for (int colCount = 0; colCount < numCols; ++colCount) {
			tempSum += inStruct->weights.values[numLayer][rowCount][colCount] * inVector[colCount];
		}
		//Sigma function
		if (tempSum < 0) {
			tempSum = 0;		//Comment this if you want to kill it.
		}
		nodesForUpdating->nodes[whichCase][numLayer][rowCount] = tempSum;	//Assumes numLayer starts at 0.
//		inStruct->layerNodes.values[numLayer + 1][rowCount] = tempSum;

	}
};


void updateLayersC(CNNStructureC * testStruct, dataC* data, reducedLayerNodesC* nodesForUpdating, int whichCase) {
	//input contains the starting layer nodes. 

//	testStruct->layerNodes.values[0] = data->inputNodes[whichCase];

	double* tempLayer;
	tempLayer = data->inputNodes[whichCase];
	for (int layerCount = 0; layerCount < testStruct->weights.numLayers; ++layerCount) {

		structMatVecMult(testStruct, nodesForUpdating, tempLayer, layerCount, whichCase);
		tempLayer = nodesForUpdating->nodes[whichCase][layerCount]; 
//		tempLayer = testStruct->layerNodes.values[layerCount + 1];
	}
}
double calcCostC(CNNStructureC* inputStruct, dataC* inputData, reducedLayerNodesC* nodesForUpdating, int whichCase, bool updateLayersBool) {
	//This version uses the input to update the layers and then calculate the cost.
	if (updateLayersBool) {
		updateLayersC(inputStruct, inputData, nodesForUpdating, whichCase);
	}
	//The cost only depends on the last layer's nodes. And there is no addition term added to the host_vector.
	double costSum = 0.;
	size_t numLayers = nodesForUpdating->numLayers;
//	size_t numLayers = inputStruct->layerNodes.numLayers;
	size_t desiredSize = inputData->numOutputs;
	for (int iCnt = 0; iCnt < desiredSize - 1; ++iCnt) {	// Cut down by one because desired has the extra 1.
															// It doesn't really matter since both have 1 at the end.
		costSum += pow((nodesForUpdating->nodes[whichCase][numLayers-1][iCnt] - inputData->outputNodes[whichCase][iCnt]), 2);
//		costSum += pow((inputStruct->layerNodes.values[numLayers - 1][iCnt] - inputData->outputNodes[whichCase][iCnt]), 2);
	}

	return(costSum);
}
void makeGradPassC(CNNStructureC* testStruct, CNNStructureC* tempGradStruct, reducedLayerNodesC* nodesForUpdating, 
	dataC* data, size_t whichCase);

void makeGradPassCUDA(CNNStructureC* testStruct, CNNStructureCFlat* tempGradStruct, reducedLayerNodesC* nodesForUpdating,
	dataC* data, size_t whichCase);

extern "C" void calcGradientPartsC(CNNStructureC holdAccumGradients[], structureC<int> *testCase, dataC* data,
	CNNStructureC *testStruct, size_t begin, double* cost, reducedLayerNodesC* nodesForUpdating, 
		const size_t numBlocks, const size_t numThreads) {

	size_t end = numBlocks * numThreads;

	for (size_t iCnt = 1; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
		setWeightsToZeros(&holdAccumGradients[iCnt]); // Zero this out because of the += later.
	}


	double tempCost = 0;
	size_t tSet;
	for (size_t blockNum = 0; blockNum < numBlocks; ++blockNum) {
		for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
			tSet = blockNum * numThreads + iCnt;
			updateLayersC(testStruct, data, nodesForUpdating, tSet);
		}
	}
	for (size_t blockNum = 0; blockNum < numBlocks; ++blockNum) {
		for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
			tSet = blockNum * numThreads + iCnt;
			tempCost += calcCostC(testStruct, data, nodesForUpdating, tSet, false); 
		}
	}
	for (size_t blockNum = 0; blockNum < numBlocks; ++blockNum) {
		for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
			tSet = blockNum * numThreads + iCnt;
// Add to holdAccumGradients for this test set. 
			makeGradPassC(testStruct, &holdAccumGradients[iCnt], nodesForUpdating, data, tSet);
		}
	}
	//For the moment, just gather them up, here.

	for (size_t iCnt = 1; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
		plusEqualsStruct(&holdAccumGradients[0], &holdAccumGradients[iCnt]);
	}

	*cost = tempCost / double(end - begin);
	printf("\nCost in calcGradientPartsC ");
	printf("\t%f", *cost);

}

extern "C" void calcGradientPartsCUDA(CNNStructureCFlat holdAccumGradients[], structureC<int> *testCase, dataC* data,
	CNNStructureC *testStruct, size_t begin, double* cost, reducedLayerNodesC* nodesForUpdating,
	const size_t numBlocks, const size_t numThreads) {

	size_t end = numBlocks * numThreads;

// Set up (zero out) numThreads number of holdAccumGradients. Each will be added to as you work towards
// the average over all the training sets. 

	for (size_t iCnt = 1; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
		setWeightsToZeros(&holdAccumGradients[iCnt]); // Zero this out because of the += later.
	}
//	launchCUDAGradientsZero(d_holdAccumGradientsCFlatWeights);

	double tempCost = 0;
	size_t tSet;
	for (size_t blockNum = 0; blockNum < numBlocks; ++blockNum) {
		for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
			tSet = blockNum * numThreads + iCnt;
			updateLayersC(testStruct, data, nodesForUpdating, tSet);
		}
	}
	for (size_t blockNum = 0; blockNum < numBlocks; ++blockNum) {
		for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
			tSet = blockNum * numThreads + iCnt;
			tempCost += calcCostC(testStruct, data, nodesForUpdating, tSet, false);
		}
	}
	for (size_t blockNum = 0; blockNum < numBlocks; ++blockNum) {
		for (size_t iCnt = 0; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
			tSet = blockNum * numThreads + iCnt;
			// Add to holdAccumGradients for this test set. 
			makeGradPassCUDA(testStruct, &holdAccumGradients[iCnt], nodesForUpdating, data, tSet);
		}
	}
	//For the moment, just gather them up, here.

	for (size_t iCnt = 1; iCnt < numThreads; ++iCnt) {	//Note, currently hardwiring in begin at 0
		plusEqualsStruct(&holdAccumGradients[0], &holdAccumGradients[iCnt]);
	}

	*cost = tempCost / double(end - begin);
	printf("\nCost in calcGradientPartsCUDA ");
	printf("\t%f", *cost);

}
void makeGradPassC(CNNStructureC* testStruct, CNNStructureC* tempGradStruct, reducedLayerNodesC* nodesForUpdating,
	dataC* data, size_t whichCase){
	// The goal here is to create the gradient for the single test case. 
	// There are multiple terms that need to be multiplied
	// together to form each element. Complete a layer (from back to front) 
	// before proceding to the next layer. The reason is that you need the results of layer L
	// inorder to get a cost for L-1.

/*Memory consideration: I have three vectors (pCPA, partRelu, and temppCpA that will be taking on varying size
as I go through the layers. Rather than resizing the vectors, I'm just going to allocate the memory for the largest
value that they will take on. Note, that I could define these once outside of here. However, at this point in my understanding, 
I think that would introduce a problem when going highly parallel, i.e., each thread should have its own copy. 
A likely more efficient way would be to make an array of each of these, then send one each into this function. Ideally,
you could leave them in place (not have to move them on, off, or create on the device each time you run this.*/
/*In going parallel, I now get the first layer nodes from the data input nodes. Then, the rest of the nodes (hidden layers and 
output nodes) come from the reducedForUpdate nodes. */
//Find the largest size vector (Note, that is a number that could be precalculated and sent in).

	size_t maxNodes = 0;
	for (size_t iCnt = 0; iCnt < testStruct->layerNodes.numLayers; ++iCnt) {
		if (testStruct->layerNodes.numNodes[iCnt] > maxNodes) {
			maxNodes = testStruct->layerNodes.numNodes[iCnt];
		}
	}
	double* pCpA = (double*)malloc(maxNodes* sizeof(double));
	double* partRelu = (double*)malloc(sizeof(double)*maxNodes);
	double* temppCpA = (double*)malloc(sizeof(double)*maxNodes);

	size_t startLayer = testStruct->layerNodes.numLayers - 1;

	for (size_t iCnt = 0; iCnt < testStruct->layerNodes.numNodes[startLayer]; ++iCnt) {
		pCpA[iCnt] = 2.*(nodesForUpdating->nodes[whichCase][startLayer-1][iCnt] - data->outputNodes[whichCase][iCnt]);
//		pCpA[iCnt] = 2.*(testStruct->layerNodes.values[startLayer][iCnt] - desired[iCnt]);
	}

	for (size_t layerCount = startLayer; layerCount > 0; --layerCount) {
		if (layerCount == 1) {
			matVecMultC(testStruct->weights.values[layerCount - 1],
				data->inputNodes[whichCase], testStruct->weights.numRows[layerCount - 1],
				testStruct->weights.numCols[layerCount - 1], partRelu);

		}
		else
		{
			matVecMultC(testStruct->weights.values[layerCount - 1],
				nodesForUpdating->nodes[whichCase][layerCount - 2], testStruct->weights.numRows[layerCount - 1],
				testStruct->weights.numCols[layerCount - 1], partRelu);
		}
//The implication of these next lines would seem to be that you do not use the final layer in calculating partRelu. 
//I have carried that into the above. 
//			matVecMultC(testStruct->weights.values[layerCount - 1],
//			testStruct->layerNodes.values[layerCount - 1], testStruct->weights.numRows[layerCount - 1],
//			testStruct->weights.numCols[layerCount - 1], partRelu);

		//Sigma
		for (size_t rowCount = 0; rowCount < testStruct->weights.numRows[layerCount - 1] - 1; ++rowCount) {

			if (partRelu[rowCount] < 0.) {
				partRelu[rowCount] = 0.;
			}
			else {
				partRelu[rowCount] = 1.;
			}
			//			partRelu[rowCount] = 1.;	//uncomment here and comment above to Kill sigma till you understand it.

			for (size_t colCount = 0; colCount < tempGradStruct->weights.numCols[layerCount - 1] - 1; ++colCount) {
				//(partial z wrt w)*partial relu*pCpA
				if (layerCount == 1) {
					tempGradStruct->weights.values[layerCount - 1][rowCount][colCount] +=
						data->inputNodes[whichCase][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}
				else {
					tempGradStruct->weights.values[layerCount - 1][rowCount][colCount] +=
						nodesForUpdating->nodes[whichCase][layerCount - 2][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}

//				tempGradStruct->weights.values[layerCount - 1][rowCount][colCount] +=
//					testStruct->layerNodes.values[layerCount - 1][colCount] * partRelu[rowCount] * pCpA[rowCount];
			}
			// Each row also has a bias term at the end of the row.
			tempGradStruct->weights.values[layerCount - 1][rowCount][testStruct->weights.numCols[layerCount - 1] - 1] +=
				partRelu[rowCount] * pCpA[rowCount];
		}
		if (layerCount > 1) {

			//Calculate the pCpA host_vector for the next round.
			for (size_t colCount = 0; colCount < testStruct->weights.numCols[layerCount - 1] - 1; ++colCount) {
				double tempSum = 0.;
				for (size_t rowCount = 0; rowCount < testStruct->weights.numRows[layerCount - 1] - 1; ++rowCount) {
					tempSum += testStruct->weights.values[layerCount - 1][rowCount][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}
				temppCpA[colCount] = tempSum;
			}
			for (size_t colCount = 0; colCount < testStruct->weights.numCols[layerCount - 1] - 1; ++colCount) {
				pCpA[colCount] = temppCpA[colCount];
			}
		}
	}
	free(pCpA);
	free(partRelu);
	free(temppCpA);
}

void makeGradPassCUDA(CNNStructureC* testStruct, CNNStructureCFlat* tempGradStruct, reducedLayerNodesC* nodesForUpdating,
	dataC* data, size_t whichCase) {
	// The goal here is to create the gradient for the single test case. 
	// There are multiple terms that need to be multiplied
	// together to form each element. Complete a layer (from back to front) 
	// before proceding to the next layer. The reason is that you need the results of layer L
	// inorder to get a cost for L-1.

/*Memory consideration: I have three vectors (pCPA, partRelu, and temppCpA that will be taking on varying size
as I go through the layers. Rather than resizing the vectors, I'm just going to allocate the memory for the largest
value that they will take on. Note, that I could define these once outside of here. However, at this point in my understanding,
I think that would introduce a problem when going highly parallel, i.e., each thread should have its own copy.
A likely more efficient way would be to make an array of each of these, then send one each into this function. Ideally,
you could leave them in place (not have to move them on, off, or create on the device each time you run this.*/
/*In going parallel, I now get the first layer nodes from the data input nodes. Then, the rest of the nodes (hidden layers and
output nodes) come from the reducedForUpdate nodes. */
//Find the largest size vector (Note, that is a number that could be precalculated and sent in).
	size_t indexFlatten;
	size_t maxNodes = 0;
	for (size_t iCnt = 0; iCnt < testStruct->layerNodes.numLayers; ++iCnt) {
		if (testStruct->layerNodes.numNodes[iCnt] > maxNodes) {
			maxNodes = testStruct->layerNodes.numNodes[iCnt];
		}
	}
	double* pCpA = (double*)malloc(maxNodes * sizeof(double));
	double* partRelu = (double*)malloc(sizeof(double)*maxNodes);
	double* temppCpA = (double*)malloc(sizeof(double)*maxNodes);

	size_t startLayer = testStruct->layerNodes.numLayers - 1;

	for (size_t iCnt = 0; iCnt < testStruct->layerNodes.numNodes[startLayer]; ++iCnt) {
		pCpA[iCnt] = 2.*(nodesForUpdating->nodes[whichCase][startLayer - 1][iCnt] - data->outputNodes[whichCase][iCnt]);
		//		pCpA[iCnt] = 2.*(testStruct->layerNodes.values[startLayer][iCnt] - desired[iCnt]);
	}

	for (size_t layerCount = startLayer; layerCount > 0; --layerCount) {
		if (layerCount == 1) {
			matVecMultC(testStruct->weights.values[layerCount - 1],
				data->inputNodes[whichCase], testStruct->weights.numRows[layerCount - 1],
				testStruct->weights.numCols[layerCount - 1], partRelu);

		}
		else
		{
			matVecMultC(testStruct->weights.values[layerCount - 1],
				nodesForUpdating->nodes[whichCase][layerCount - 2], testStruct->weights.numRows[layerCount - 1],
				testStruct->weights.numCols[layerCount - 1], partRelu);
		}
		//The implication of these next lines would seem to be that you do not use the final layer in calculating partRelu. 
		//I have carried that into the above. 
		//			matVecMultC(testStruct->weights.values[layerCount - 1],
		//			testStruct->layerNodes.values[layerCount - 1], testStruct->weights.numRows[layerCount - 1],
		//			testStruct->weights.numCols[layerCount - 1], partRelu);

				//Sigma
		for (size_t rowCount = 0; rowCount < testStruct->weights.numRows[layerCount - 1] - 1; ++rowCount) {

			if (partRelu[rowCount] < 0.) {
				partRelu[rowCount] = 0.;
			}
			else {
				partRelu[rowCount] = 1.;
			}
			//			partRelu[rowCount] = 1.;	//uncomment here and comment above to Kill sigma till you understand it.

			for (size_t colCount = 0; colCount < tempGradStruct->weights.numCols[layerCount - 1] - 1; ++colCount) {
				//(partial z wrt w)*partial relu*pCpA
				if (layerCount == 1) {
					indexFlatten = tempGradStruct->weights.startLayer[layerCount - 1] + 
						tempGradStruct->weights.numCols[layerCount-1]*rowCount + colCount;

					tempGradStruct->weights.values[indexFlatten] +=
						data->inputNodes[whichCase][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}
				else {
					indexFlatten = tempGradStruct->weights.startLayer[layerCount - 1] +
						tempGradStruct->weights.numCols[layerCount - 1] * rowCount + colCount;
					tempGradStruct->weights.values[indexFlatten] +=
						nodesForUpdating->nodes[whichCase][layerCount - 2][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}

				//				tempGradStruct->weights.values[layerCount - 1][rowCount][colCount] +=
				//					testStruct->layerNodes.values[layerCount - 1][colCount] * partRelu[rowCount] * pCpA[rowCount];
			}
			// Each row also has a bias term at the end of the row.
			indexFlatten = tempGradStruct->weights.startLayer[layerCount - 1] +
				tempGradStruct->weights.numCols[layerCount - 1] * rowCount + testStruct->weights.numCols[layerCount - 1] - 1;
			tempGradStruct->weights.values[indexFlatten] += partRelu[rowCount] * pCpA[rowCount];
		}
		if (layerCount > 1) {

			//Calculate the pCpA host_vector for the next round.
			for (size_t colCount = 0; colCount < testStruct->weights.numCols[layerCount - 1] - 1; ++colCount) {
				double tempSum = 0.;
				for (size_t rowCount = 0; rowCount < testStruct->weights.numRows[layerCount - 1] - 1; ++rowCount) {
					tempSum += testStruct->weights.values[layerCount - 1][rowCount][colCount] * partRelu[rowCount] * pCpA[rowCount];
				}
				temppCpA[colCount] = tempSum;
			}
			for (size_t colCount = 0; colCount < testStruct->weights.numCols[layerCount - 1] - 1; ++colCount) {
				pCpA[colCount] = temppCpA[colCount];
			}
		}
	}
	free(pCpA);
	free(partRelu);
	free(temppCpA);
}



