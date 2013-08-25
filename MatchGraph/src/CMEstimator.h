/*
 * CMEstimator.h
 *
 * Interface for estimator module (finding k-best confidence measures).
 *
 *  Created on: 29.05.2013
 *      Author: Fabian
 */

#ifndef CMESTIMATOR_H_
#define CMESTIMATOR_H_

#include "MatrixHandler.h"

class CMEstimator {
public:
	virtual ~CMEstimator(){};

	//Function to fill the two index arrays with the best image-pairs that should be compared
	virtual void getKBestConfMeasures(MatrixHandler* T, float* F, int kBest) = 0;

	//Function to fill the two index arrays with random image-pairs (random iteration)
	virtual void computeRandomComparisons(MatrixHandler* T, const int k) = 0;

	//Resulting index array1 (containing i-th index)
	virtual int* getIdx1Ptr() = 0;

	//Resulting index array2 (containing j-th index)
	virtual int* getIdx2Ptr() = 0;

	//Array for the image-comparison result
	virtual int* getResPtr() = 0;
};

#endif /* CMESTIMATOR_H_ */
