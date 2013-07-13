/*
 * CMEstimatorGPUApprox.h
 *
 *  Created on: 05.06.2013
 *      Author: furby
 */

#ifndef CMESTIMATORGPUAPPROX_H_
#define CMESTIMATORGPUAPPROX_H_

#include "CMEstimator.h"

class CMEstimatorGPUApprox : public CMEstimator{
public:
	CMEstimatorGPUApprox();
	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	Indices* getInitializationIndices(MatrixHandler* T, int initNr);
	//TODO destructor
	//virtual ~CMEstimatorGPUApprox();
};

#endif /* CMESTIMATORGPUAPPROX_H_ */
