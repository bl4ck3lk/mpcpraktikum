/*
 * CMEstimatorCPU.h
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#ifndef CMESTIMATORCPU_H_
#define CMESTIMATORCPU_H_

#include "CMEstimator.h"

class CMEstimatorCPU : public CMEstimator{
public:
	CMEstimatorCPU();
	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	Indices* getInitializationIndices(MatrixHandler* T, int initNr);
	//todo destructor
	//virtual ~CMEstimatorCPU();
};

#endif /* CMESTIMATORCPU_H_ */
