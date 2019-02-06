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

	for (int rowCount = 0; rowCount < v.size(); ++rowCount) {
			os << v[rowCount];
	}
//	os << "\n";
	return os;
}

// Template to print vector of vectors container elements 
template <typename T>
ostream& operator<<(ostream& os, const vector<vector<vector<T>>>& v)
{

	for (int layerCount = 0; layerCount < v.size(); ++layerCount) {
		os << v[layerCount];
	}
//	os << "\n";
	return os;
}

// Template to multiply a vector (left side) by a matrix (right side)
template <typename T>
vector<T> vecMatMult(const vector<T>& inVec, const vector<vector<T>>& inMat) {
	vector <T> tempVec;
	tempVec.reserve(inMat[0].size());
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

// Template to multiply a matrix (left side) by a vector (right side)
template <typename T>
vector<T> matVecMult(const vector<vector<T>>& inMat,const vector<T>& inVec ) {
	vector <T> tempVec;
	assert(inVec.size() == inMat[0].size());
	tempVec.reserve(inMat.size());
	for (int rowCount = 0; rowCount < inMat.size(); rowCount++) {
		double tempSum = 0;
		for (int indexCount = 0; indexCount < inVec.size(); indexCount++) {
			tempSum += inVec[indexCount] * inMat[rowCount][indexCount];
		}
		tempVec.push_back(tempSum);
	}
	return(tempVec);
}
template <typename T>
vector<T> subtract(const vector<T>& rhs, const vector<T>& lhs) {
	assert(lhs.size() == rhs.size());
	vector<T> tempOut;
	tempOut.reserve(lhs.size());
	for (size_t iCnt = 0; iCnt < lhs.size(); ++iCnt) {
		tempOut.push_back(lhs[iCnt] - rhs[iCnt]);
	}
	return(tempOut);	//Send back a copy.
}

// Template to overload subtraction of one vector from another. 
template <typename T>
vector<T> operator-(const vector<T>& lhs, const vector<T>& rhs)
{
	assert(lhs.size() == rhs.size());
	vector<T> tempVec;
	tempVec.reserve(rhs.size());
	for (int rowCount = 0; rowCount < rhs.size(); ++rowCount) {
		tempVec.push_back(lhs[rowCount] - rhs[rowCount]);
	}
	return tempVec;
}
// Template to overload scaler multiplication of a vector. 
template <typename T>
vector<T> operator*(double scaler, const vector<T>& rhs)
{
	vector<T> tempVec;
	tempVec.reserve(rhs.size());
	for (int rowCount = 0; rowCount < rhs.size(); ++rowCount) {
		tempVec.push_back(scaler*rhs[rowCount]);
	}
	return tempVec;
}

#endif