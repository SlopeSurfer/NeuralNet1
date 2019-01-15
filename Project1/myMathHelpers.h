#ifndef myMathHelpers_H
#define myMathHelpers_H

#include <iostream>
#include <vector>
#include <assert.h>
using namespace std;
// Template to print vector container elements 
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	os << "[";
	for (int i = 0; i < v.size(); ++i) {
		os << v[i];
		if (i != v.size() - 1)
			os << ", ";
	}
	os << "]\n";
	return os;
}
// Template to print vector of vectors container elements 
template <typename T>
ostream& operator<<(ostream& os, const vector<vector<T>>& v)
{
	os << "\n";
	for (int rowCount = 0; rowCount < v.size(); ++rowCount) {
			os << v[rowCount];
	}
	return os;
}

// Template to multiply a vector by a matrix
template <typename T>
vector<T> vecMatMult(vector<T> inVec, vector<vector<T>> inMat) {
	vector <T> tempVec;
	assert(inVec.size() == inMat.size());

	for (int colCount = 0; colCount < inMat[0].size(); colCount++) {
		double tempSum = 0;
		for (int indexCount = 0; indexCount < inVec.size(); indexCount++) {
			tempSum += inVec[indexCount] * inMat[indexCount][colCount];
		}
		tempVec.push_back(tempSum);
	}
	return(tempVec);
}
#endif