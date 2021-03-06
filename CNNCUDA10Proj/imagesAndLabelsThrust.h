#ifndef IMAGESANDLABELS_H
#define IMAGESANDLABELS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <thrust\host_vector.h>
#include <string>	//Include this not to get string, but to get operators like << on string. 
using namespace std;
using namespace thrust;

class imagesAndLabelsThrust
{
	host_vector<host_vector<host_vector<double>>> images;	//host_vector of 2D images
	host_vector<double> labels;
	int numImages;
	int numRows, numCols;
public:
	imagesAndLabelsThrust(string fileNameImages, string fileNameLabels);
	void displayImage(int imageNum);
	size_t getPixel(int imageNumber, int row, int col);
	size_t getLabel(int imageNumber);
	int getNumImages();
	int getNumRows();
	int getNumCols();
	friend class handNumberData;
};

#endif