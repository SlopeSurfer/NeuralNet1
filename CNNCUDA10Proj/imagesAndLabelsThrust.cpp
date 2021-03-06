#include "imagesAndLabelsThrust.h"
#include <assert.h>

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

imagesAndLabelsThrust::imagesAndLabelsThrust(string fileNameImages, string fileNameLabels)
{
//Get the images.
	ifstream fileImages(fileNameImages, ios::binary);
	if (fileImages.is_open())
	{
		int magicNumber = 0;
		fileImages.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = ReverseInt(magicNumber);
		fileImages.read((char*)&numImages, sizeof(numImages));
		numImages = ReverseInt(numImages);
		fileImages.read((char*)&numRows, sizeof(numRows));
		numRows = ReverseInt(numRows);
		fileImages.read((char*)&numCols, sizeof(numCols));
		numCols = ReverseInt(numCols);
#ifdef _DEBUG
		cout << "\n\nImages data ";
		cout << "\nMagic number " << magicNumber;
		cout << "\nNumber of rows " << numRows;
		cout << "\nNumber of columns " << numCols;
		cout << "\nNumber of images " << numImages;
#endif
		for (int i = 0; i < numImages; ++i)
		{
			vector<vector<double>> tp;
			for (int r = 0; r < numRows; ++r)
			{
				vector<double> tpp;
				for (int c = 0; c < numCols; ++c)
				{
					unsigned char temp = 0;
					fileImages.read((char*)&temp, sizeof(temp));
					tpp.push_back((double)temp);
				}
				tp.push_back(tpp);
			}
			images.push_back(tp);
		}
		fileImages.close();
	}
	else {
		cout << "\nCould not open file " << fileNameImages;
	}
//Get the labels
	ifstream fileLabels(fileNameLabels, ios::binary);
	if (fileLabels.is_open())
	{
		int magicNumber = 0;
		int numLabels = 0;
		fileLabels.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = ReverseInt(magicNumber);

		fileLabels.read((char*)&numLabels, sizeof(numLabels));
		numLabels = ReverseInt(numLabels);

		assert(numLabels == numImages);
#ifdef _DEBUG
		cout << "\n\nLabels data ";
		cout << "\nMagic number " << magicNumber;
		cout << "\nNumber labels " << numLabels;
#endif
		for (int i = 0; i < numLabels; ++i)
		{
			unsigned char temp = 0;
			fileLabels.read((char*)&temp, sizeof(temp));
			labels.push_back(temp);

		}
		fileLabels.close();
	}
	else {
		cout << "\nCould not open file " << fileNameImages;

	}
}

void imagesAndLabelsThrust::displayImage(int imageToCheck){
	assert(imageToCheck > 0 && imageToCheck <= numImages);
	cout << "\nLabel at " << imageToCheck << " = " << labels[imageToCheck];
	for (int iCnt = 0; iCnt < numRows; iCnt++) {
		cout << "\n";
		for (int jCnt = 0; jCnt < numCols; jCnt++) {
			if (images[imageToCheck] [iCnt] [jCnt]) {
				cout << " " << " ";
			}
			else
			{
				cout << " " << 0;
			}
		}
	}
}
size_t imagesAndLabelsThrust::getPixel(int imageNumber, int row, int col) {
	return(images[imageNumber] [row] [col]);
}

size_t imagesAndLabelsThrust::getLabel(int imageNumber) {
	return(labels[imageNumber]);
}

int imagesAndLabelsThrust::getNumImages() {
	return(numImages);
}
int imagesAndLabelsThrust::getNumRows() {
	return(numRows);

}int imagesAndLabelsThrust::getNumCols() {
	return(numCols);
}