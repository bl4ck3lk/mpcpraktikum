/*
 * CMEstimator.h
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#ifndef CMESTIMATOR_H_
#define CMESTIMATOR_H_

#include "MatrixHandler.h"

class CMEstimator {
public:
	virtual ~CMEstimator(){};
	virtual void getKBestConfMeasures(MatrixHandler* T, float* F, int kBest) = 0;
	virtual void computeRandomComparisons(MatrixHandler* T, const int k) = 0;
	virtual int* getIdx1Ptr() = 0;
	virtual int* getIdx2Ptr() = 0;
	virtual int* getResPtr() = 0;
};

#endif /* CMESTIMATOR_H_ */
