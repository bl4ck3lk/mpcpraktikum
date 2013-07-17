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
	~CMEstimatorGPUSparse();

	int* getIdx1Ptr();
	int* getIdx2Ptr();
	int* getResPtr();

	void getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	void computeRandomComparisons(MatrixHandler* T, const int k);

private:
	int lastSize;

	int* d_idx1;
	int* d_idx2;
	int* d_res;

	int* idx1; //todo only for testing purpose
	int* res; //todo only for testing purpose

	/* cula */

	int determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, int dim, int kBest, int kBestForThisColumn, int currIndexNr);
	culaSparseStatus computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, double* A, int* rowPtr, int* colIdx, double* x, double* b);
	void initIdxDevicePointers(int size, unsigned int dim);
	void initCula();
};

#endif /* CMESTIMATORGPUSPARSE_H_ */
