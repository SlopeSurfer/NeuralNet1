#ifndef handNumberData_H
#define handNumberData_H
#include "dataSet.h"
class handNumberData :
	public dataSet
{
	size_t numCols;	//Used to reconstruct the images for display.

public:
	handNumberData(const string& imageDataName, const string& labelDataName);
	~handNumberData();
	void displayImage(const size_t& index);
};

#endif
