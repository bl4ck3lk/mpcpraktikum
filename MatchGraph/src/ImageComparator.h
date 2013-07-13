/*
 * ImageComparator.h
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#ifndef IMAGECOMPARATOR_H_
#define IMAGECOMPARATOR_H_

#include "MatrixHandler.h"
#include "CMEstimator.h"
#include "ImageHandler.h"

class ImageComparator {
public:
	virtual void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize) = 0;
	virtual void setRandomMode(bool mode) = 0;
	virtual ~ImageComparator(){};
};

#endif /* IMAGECOMPARATOR_H_ */
