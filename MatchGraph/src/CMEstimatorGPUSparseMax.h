/*
 * CMEstimatorGPUSparseMax.h
 *
 *  Created on: Jul 16, 2013
 *      Author: schwarzk
 */

#ifndef CMESTIMATORGPUSPARSEMAX_H_
#define CMESTIMATORGPUSPARSEMAX_H_

#include "CMEstimator.h"
#include <stdlib.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <ctime>
#include <cula_sparse.h>

class CMEstimatorGPUSparseMax : public CMEstimator{
public:
	CMEstimatorGPUSparseMax();
	~CMEstimatorGPUSparseMax();

	int* getIdx1Ptr();
	int* getIdx2Ptr();
	int* getResPtr();

	int* getIdx1HostPtr(int dim); //todo only for testing purpose
	int* getResHostPtr(int dim); //todo only for testing purpose
	void setResDevicePtr(int* res, int dim); //todo only for testing purpose

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

	void determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, long* d_tmpIndices, double* d_tmpConfidences, int dim, int kBest);
	culaSparseStatus computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, double* A, int* rowPtr, int* colIdx, double* x, double* b);
	void initIdxDevicePointers(int size, unsigned int dim);
};

#endif /* CMESTIMATORGPUSPARSEMAX_H_ */
