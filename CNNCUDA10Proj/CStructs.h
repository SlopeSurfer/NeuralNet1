#ifndef CSTRUCTS_H
#define CSTRUCTS_H
#include <stdio.h>

struct weightsC {
	double*** values;
	int numLayers;
	int* numRows;
	int* numCols;
};

struct layerNodesC {
	double** values;
	int numLayers;
	int* numNodes;
};

struct reducedLayerNodesC {
	size_t numSets;
	size_t numLayers;
	size_t* nodeLengths;
	double*** nodes;
};

struct CNNStructureC {
	weightsC weights;
	layerNodesC layerNodes;
};
template <class myType>
struct structureC {
	myType* structure;
	int numElements;
};

struct dataC {
	double** inputNodes;
	double** outputNodes;
	int * labels;
	int numInputs;
	int numOutputs;
	int numSets;
};

class Managed {
public:
	void *operator new(size_t len) {
		void *ptr;
//		cout << "\nCalling overloaded new operator ";
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void *ptr) {
//		cout << "\nCalling overloaded delete operator ";
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

class SimpleMatrix : public Managed {
public:
	double* matrix;
	size_t numRows;
	size_t numCols;

	SimpleMatrix(size_t rows, size_t cols, double value = 0) : numRows(rows), numCols(cols) {
//		printf("\nCalling SimpleMatrix constructor ");
		cudaMallocManaged(&matrix, numRows*numCols*sizeof(double));
		for (size_t row = 0; row < numRows; ++row) {
			for (size_t col = 0; col < numCols; ++col) {
				matrix[numCols*row + col] = value;
			}
		}
	}
	SimpleMatrix(const SimpleMatrix & copy) {
		printf("\nCalling SimpleMatrix copy constructor ");
		numRows = copy.numRows;
		numCols = copy.numCols;
		cudaMallocManaged(&matrix, numRows*numCols*sizeof(double));
		for (size_t row = 0; row < numRows; ++row) {
			for (size_t col = 0; col < numCols; ++col) {
				matrix[numCols*row + col] = copy.matrix[numCols*row + col];
			}
		}
	}

	SimpleMatrix& operator =(const SimpleMatrix & rhs) {
		//Using copy and swap method. It is supposed to be safer.
		printf("\nCalling SimpleMatrix assignment operator ");

		if (this != &rhs) {
			SimpleMatrix tmp(rhs);
			//Now, swap the data members with the temporary.
			std::swap(numRows, tmp.numRows);
			std::swap(numCols, tmp.numCols);
			std::swap(matrix, tmp.matrix);
		}
		return(*this);
	}

	double getMatrixElem(size_t row, size_t col) {
		return(matrix[numCols*row + col]);
	}

	void setMatrixElem(size_t row, size_t col, double value) {
		matrix[numCols*row + col] = value;
	}

	double* getWholeMatrix() {
		return(matrix);
	}
	size_t getNumRows() {
		return(numRows);
	}
	size_t getNumCols() {
		return(numCols);
	}

};
#endif
