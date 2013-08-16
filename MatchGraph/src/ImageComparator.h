/*
 * ImageComparator.h
 *
 * Interface for the comparison module.
 *
 *  Created on: May 29, 2013
 *      Author: Armin, Fabian
 */

#ifndef IMAGECOMPARATOR_H_
#define IMAGECOMPARATOR_H_

#include "MatrixHandler.h"
#include "CMEstimator.h"
#include "ImageHandler.h"

class ImageComparator {
public:
	virtual ~ImageComparator(){};

	//Function to execute the actual image comparison on the given image-pairs and storing the result in the given memory location
	virtual void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* idx1, int* idx2, int* res, int arraySize) = 0;

	//Function to switch off the actual image comparison, such that the result array is filled randomly
	virtual void setRandomMode(bool mode) = 0;
};

#endif /* IMAGECOMPARATOR_H_ */
