/*
 * CMEstimatorCPU.h
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#ifndef CMESTIMATORCPU_H_
#define CMESTIMATORCPU_H_

#include "CMEstimator.h"
#include "../lib/Eigen/Eigen/Dense"

class CMEstimatorCPU : public CMEstimator{
public:
	CMEstimatorCPU();

	void getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	void computeRandomComparisons(MatrixHandler* T, const int k);

	~CMEstimatorCPU();

	int* getIdx1Ptr();
	int* getIdx2Ptr();
	int* getResPtr();

private:
	int* idx1;
	int* idx2;
	int* res;
	int currentArraySize;

	void initIdxArrays(int arraySize, int dim);
};

#endif /* CMESTIMATORCPU_H_ */
