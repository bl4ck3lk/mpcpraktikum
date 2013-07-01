/*
 * CMEstimatorGPUSparse.h
 *
 *  Created on: 19.06.2013
 *      Author: Fabian
 */

#ifndef CMESTIMATORGPUSPARSE_H_
#define CMESTIMATORGPUSPARSE_H_

#include "CMEstimator.h"
#include <stdlib.h>
#include <cula_sparse.h>

class CMEstimatorGPUSparse : public CMEstimator{
public:
	CMEstimatorGPUSparse();
	virtual ~CMEstimatorGPUSparse();
	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest); //todo change return value to void
	Indices* getInitializationIndices(MatrixHandler* T, int initNr);

private:
	int lastSize;

	int* d_idx1;
	int* d_idx2;
	int* d_res;

	Indices* getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest); //TODO remove me
	int determineBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest, int kBestForThisColumn, int currIndexNr);
	void computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, float* A, int* rowPtr, int* colIdx, float* x, float* b);
	void initIdxDevicePointers(int size, unsigned int dim);
};

#endif /* CMESTIMATORGPUSPARSE_H_ */
