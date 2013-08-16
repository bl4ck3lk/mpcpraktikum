/*
 * CMEstimatorCPU.h
 *
 * Header file for a CPU estimator, implementing the CMEstimator interface.
 *
 *  Created on: 29.05.2013
 *      Author: Fabian
 */

#ifndef CMESTIMATORCPU_H_
#define CMESTIMATORCPU_H_

#include "CMEstimator.h"
#include "../lib/Eigen/Eigen/Dense"

class CMEstimatorCPU : public CMEstimator{
public:
	CMEstimatorCPU();
	~CMEstimatorCPU();

	//Implemented abstract functions (see CMEstimator.h)
	void getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	void computeRandomComparisons(MatrixHandler* T, const int k);
	int* getIdx1Ptr();
	int* getIdx2Ptr();
	int* getResPtr();

private:
	//Host memory for resulting arrays managed by the specific estimator
	int* idx1;
	int* idx2;
	int* res;

	//Current size of resulting arrays
	int currentArraySize;

	//Allocate resulting arrays on host
	void initIdxArrays(int arraySize, int dim);
};

#endif /* CMESTIMATORCPU_H_ */
