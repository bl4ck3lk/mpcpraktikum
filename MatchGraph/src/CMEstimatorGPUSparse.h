/*
 * CMEstimatorGPUSparse.h
 *
 *  Created on: 19.06.2013
 *      Author: Fabian
 */

#ifndef CMESTIMATORGPUSPARSE_H_
#define CMESTIMATORGPUSPARSE_H_

#include "CMEstimator.h"
#include <cula_sparse.h>

class CMEstimatorGPUSparse : public CMEstimator{
public:
	CMEstimatorGPUSparse();
	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	Indices* getInitializationIndices(MatrixHandler* T, int initNr);
	//TODO destructor
	//virtual ~CMEstimatorGPUSparse();

private:
	Indices* getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest);
	void computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, float* A, int* rowPtr, int* colIdx, float* x, float* b);
};

#endif /* CMESTIMATORGPUSPARSE_H_ */
