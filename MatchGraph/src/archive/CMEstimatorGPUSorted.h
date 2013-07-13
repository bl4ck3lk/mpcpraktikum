/*
 * CMEstimatorGPUSorted.h
 *
 *  Created on: 30.05.2013
 *      Author: Fabian
 */

#ifndef CMESTIMATORGPUSORTED_H_
#define CMESTIMATORGPUSORTED_H_

#include "CMEstimator.h"

class CMEstimatorGPUSorted : public CMEstimator {
public:
	CMEstimatorGPUSorted();
	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	Indices* getInitializationIndices(MatrixHandler* T, int initNr);
	//todo destructor
	//virtual ~CMEstimatorGPUSorted();
};

#endif /* CMESTIMATORGPUSORTED_H_ */
