#include <iostream>
#include <fstream>
#include <sstream>
//#include <opencv2/opencv.hpp>

#include <vector>
#include <string>	//Include this not to get string, but to get operators like << on string. 
#include "imagesAndLabels.h"

using namespace std;

int main() {
	cout << "Hello World!";

	string fileNameLabels = "./data/t10k-labels.idx1-ubyte";
	string fileNameImages = "./data/t10k-images.idx3-ubyte";

	imagesAndLabels testImages(fileNameImages, fileNameLabels);

	int imageToCheck = 1;
	cout << "\nLabel at " <<imageToCheck<< testImages.getLabel(imageToCheck);
	cout << "\nImage at " << imageToCheck<< endl;

	while (imageToCheck > 0 && imageToCheck < testImages.getNumImages()) {
		cout << "\nLabel at " << imageToCheck<<" = " << testImages.getLabel(imageToCheck);
		for (int iCnt = 0; iCnt < testImages.getNumRows(); iCnt++) {
			cout << "\n";
			for (int jCnt = 0; jCnt < testImages.getNumCols(); jCnt++) {
				if (testImages.getPixel(imageToCheck, iCnt, jCnt)) {
					cout << " " << " ";
				}
				else
				{
					cout << " " << 0;
				}
//				cout << " " << testImages.getPixel(imageToCheck, iCnt, jCnt);
			}
		}
		cout << "\nEnter an image number";
		cin >> imageToCheck;
	}

	return 0;
}