#ifndef IMAGESANDLABELS_H
#define IMAGESANDLABELS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
using namespace std;

class imagesAndLabels
{
	vector<vector<vector<double>>> images;	//Vector of 2D images
	vector<double> labels;
	int numImages;
	int numRows, numCols;
public:
	imagesAndLabels(string fileNameImages, string fileNameLabels);
	~imagesAndLabels();
	void displayImage(int imageNum);
	double getPixel(int imageNumber, int row, int col);
	double getLabel(int imageNumber);
	int getNumImages();
	int getNumRows();
	int getNumCols();
};

#endif