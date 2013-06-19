/*
 * CMEstimatorGPUColumn.h
 *
 *  Created on: 19.06.2013
 *      Author: Fabian
 */

#ifndef CMESTIMATORGPUCOLUMN_H_
#define CMESTIMATORGPUCOLUMN_H_

#include "CMEstimator.h"

class CMEstimatorGPUColumn : public CMEstimator{
public:
	CMEstimatorGPUColumn();
	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	Indices* getInitializationIndices(MatrixHandler* T, int initNr);
	Indices* getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest);
	//TODO destructor
	//virtual ~CMEstimatorGPUColumn();
};

#endif /* CMESTIMATORGPUCOLUMN_H_ */
