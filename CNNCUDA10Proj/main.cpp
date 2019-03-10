#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include <thread>
#include "imagesAndLabels.h"
#include "CNNStructureThrust.h"
#include "myMathHelpersThrust.h"
#include "dataSet.h"
#include "handNumberData.h"
#include <unordered_map>
#include "limits.h"
#include <chrono>
#include <thrust\host_vector.h>
using namespace thrust;
//The plan is to use thrust for the host side vectors and then convert to c style vectors when using CUDA kernals. 
//Note, this may lead to using device side thrust vectors as well. 

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA

#include <helper_cuda.h>
extern "C" int MatrixMultiply(int argc, char **argv,
	int block_size, const dim3 &dimsA,
	const dim3 &dimsB);
/**
 * Program main Started from CUDA example simple matrix multiplication.
 */
int main(int argc, char **argv) {
	host_vector<double> testV(5);
	testV[3] = 1.;

	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
		checkCmdLineFlag(argc, (const char **)argv, "?")) {
		printf("Usage -device=n (n >= 0 for deviceID)\n");
		printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
		printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
		printf("  Note: Outer matrix dimensions of A & B matrices" \
			" must be equal.\n");

		exit(EXIT_SUCCESS);
	}

	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line
	int dev = findCudaDevice(argc, (const char **)argv);

	int block_size = 32;

	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

	// width of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
		dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
	}

	// height of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
		dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
	}

	// width of Matrix B
	if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
		dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
	}

	// height of Matrix B
	if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
		dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
	}

	std::cout << "\nargv " << argv[1] << " "<< argv[2] << " " << argv[3] << " " << argv[4];
	if (dimsA.x != dimsB.y) {
		printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
			dimsA.x, dimsB.y);
		exit(EXIT_FAILURE);
	}

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
		dimsB.x, dimsB.y);

	int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);



	exit(matrix_result);
}
