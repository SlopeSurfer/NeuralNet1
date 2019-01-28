#include "handNumberData.h"
#include "imagesAndLabels.h"
#include "myMathHelpers.h"

handNumberData::handNumberData(const string& imageDataName, const string& labelDataName)
{
	imagesAndLabels* myImagesAndLabels = new imagesAndLabels(imageDataName, labelDataName);
	myImagesAndLabels->displayImage(5);
	numCols = myImagesAndLabels->getNumCols();

//Fill the dataSet.
//The images are vectors of vectors that I need to unwrap into a single vector. Note that 
//how you unwrap the data could make a difference. I'm just going to append by adding rows at a time. 

	for (size_t nodeCount = 0; nodeCount < myImagesAndLabels->getNumImages(); ++nodeCount) {
		vector<double> tempVec;
		tempVec.reserve(myImagesAndLabels->getNumRows()*myImagesAndLabels->getNumCols());
		for (size_t lineCount = 0; lineCount < myImagesAndLabels->getNumRows(); ++lineCount) {
			tempVec.insert(tempVec.end(), myImagesAndLabels->images[nodeCount][lineCount].begin(),
				myImagesAndLabels->images[nodeCount][lineCount].end());

		}
//The image data runs 0 to 255. Knock this down.
		for (size_t iCnt = 0; iCnt < tempVec.size();++iCnt) {
			tempVec[iCnt] /= 5*25500.;
		}
//Insert the extra 1.
		tempVec.push_back(0.0001);	 //Squashing this for the same reason as the data above.
		inputNodes.push_back(tempVec);

// The output nodes for this case are vectors of length 10, with one element turned to 1 (matching
// the image label) and the rest turned to zero. They are used as the desired vector during the optimization.
// It also has the extra 1 at the end?
		labels.push_back(myImagesAndLabels->labels[nodeCount]);
		vector<double> tempVec1;
		for (size_t iCnt = 0; iCnt < labels[nodeCount]; ++iCnt) {
			tempVec1.push_back(0.);
		}
		tempVec1.push_back(1.);
		for (size_t iCnt = labels[nodeCount] +1; iCnt < 10; ++iCnt) {
			tempVec1.push_back(0.);
		}
//Add the extra 1.
		tempVec1.push_back(1.);
		outputNodes.push_back(tempVec1);
	}
	delete myImagesAndLabels;
}
handNumberData::~handNumberData()
{
}
void handNumberData::displayImage(const size_t& imageToCheck) {
//The data is assumed to be in one long vector. You know that it has columns of size numCols.
//Reconstruct and display the image. 
	size_t numRows = inputNodes[0].size() / numCols;
//	cout << "\nnumRows in image reconstruction " << numRows << " Should be 28 ";
	assert(imageToCheck > 0 && imageToCheck <= numImages);
	cout << "\nLabel at " << imageToCheck << " = " << labels[imageToCheck];
	for (int iCnt = 0; iCnt < numRows; iCnt++) {
		cout << "\n";
		for (int jCnt = 0; jCnt < numCols; jCnt++) {
			if (inputNodes[imageToCheck][iCnt*numRows+jCnt]) {
				cout << " " << " ";
			}
			else
			{
				cout << " " << 0;
			}
			//				cout << " " << testImages.getPixel(imageToCheck, iCnt, jCnt);
		}
	}
}
