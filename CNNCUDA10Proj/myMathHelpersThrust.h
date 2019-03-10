#ifndef myMathHelpers_H
#define myMathHelpers_H

#include <iostream>
#include <thrust\host_vector.h>
#include <assert.h>
using namespace std;
using namespace thrust;
// Template to print vector container elements 
template <typename T>
ostream& operator<<(ostream& os, const host_vector<T>& v)
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
// Template to print host_vector of host_vectors container elements 
template <typename T>
ostream& operator<<(ostream& os, const host_vector<host_vector<T>>& v)
{

	for (int rowCount = 0; rowCount < v.size(); ++rowCount) {
			os << v[rowCount];
	}
//	os << "\n";
	return os;
}

// Template to print host_vector of host_vectors container elements 
template <typename T>
ostream& operator<<(ostream& os, const host_vector<host_vector<host_vector<T>>>& v)
{

	for (int layerCount = 0; layerCount < v.size(); ++layerCount) {
		os << v[layerCount];
	}
//	os << "\n";
	return os;
}

// Template to multiply a host_vector (left side) by a matrix (right side)
template <typename T>
host_vector<T> vecMatMult(const host_vector<T>& inVec, const host_vector<host_vector<T>>& inMat) {
	host_vector <T> tempVec;
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

// Template to multiply a matrix (left side) by a host_vector (right side)
template <typename T>
host_vector<T> matVecMult(const host_vector<host_vector<T>>& inMat,const host_vector<T>& inVec ) {
	host_vector <T> tempVec;
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
host_vector<T> subtract(const host_vector<T>& rhs, const host_vector<T>& lhs) {
	assert(lhs.size() == rhs.size());
	host_vector<T> tempOut;
	tempOut.reserve(lhs.size());
	for (size_t iCnt = 0; iCnt < lhs.size(); ++iCnt) {
		tempOut.push_back(lhs[iCnt] - rhs[iCnt]);
	}
	return(tempOut);	//Send back a copy.
}

// Template to overload subtraction of one host_vector from another. 
template <typename T>
host_vector<T> operator-(const host_vector<T>& lhs, const host_vector<T>& rhs)
{
	assert(lhs.size() == rhs.size());
	host_vector<T> tempVec;
	tempVec.reserve(rhs.size());
	for (int rowCount = 0; rowCount < rhs.size(); ++rowCount) {
		tempVec.push_back(lhs[rowCount] - rhs[rowCount]);
	}
	return tempVec;
}
// Template to overload scaler multiplication of a host_vector. 
template <typename T>
host_vector<T> operator*(double scaler, const host_vector<T>& rhs)
{
	host_vector<T> tempVec;
	tempVec.reserve(rhs.size());
	for (int rowCount = 0; rowCount < rhs.size(); ++rowCount) {
		tempVec.push_back(scaler*rhs[rowCount]);
	}
	return tempVec;
}

#endif