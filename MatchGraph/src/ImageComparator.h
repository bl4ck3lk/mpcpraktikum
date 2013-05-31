/*
 * ImageComparator.h
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#ifndef IMAGECOMPARATOR_H_
#define IMAGECOMPARATOR_H_

#include "MatrixHandler.h"
#include "ImageHandler.h"
#include "CMEstimator.h"

//TODO probably needs imageHandler or some knowledge of actual image data

class ImageComparator {
public:
	virtual void doComparison(ImageHandler* iHandler, MatrixHandler* T, int k, Indices* kBestIndices) = 0;
	virtual ~ImageComparator(){};
};

#endif /* IMAGECOMPARATOR_H_ */
