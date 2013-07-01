/*
 * Initializer.h
 *
 *  Created on: Jun 29, 2013
 *      Author: schwarzk
 */

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include "MatrixHandler.h"
//#include "ImageComparator.h"
#include <map>
#include <string>

class ImageComparator;

class Initializer {
public:
	Initializer(const char* imgDir); //constructor
	~Initializer(); //destructor

	/* for image handling */
	const char* getImage(int img); //get specific image
	const char* getFullImagePath(int img);
	int getTotalNr(); //get total nr of images
	int getMapSize();
	void printMap();
	void sortImages();
	void fillWithEmptyImages(unsigned int num);

	/* for T Matrix initialization */
	bool doInitializationPhase(MatrixHandler* T, ImageComparator* comparator, int comparatorArraySize, int similarThreshold, int maxUpdateArraySize);

private:
	std::map<int,std::string> images;
	int nrImages; //total number of images
	std::string directory;
};

#endif /* INITIALIZER_H_ */
