/*
 * CMEstimatorGPUSparseOPT.h
 *
 *  Created on: Jul 16, 2013
 *      Author: schwarzk
 */

#ifndef CMESTIMATORGPUSPARSEOPT_H_
#define CMESTIMATORGPUSPARSEOPT_H_

#include "CMEstimator.h"
#include <stdlib.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <ctime>
#include <cula_sparse.h>

class CMEstimatorGPUSparseOPT : public CMEstimator{
public:
	CMEstimatorGPUSparseOPT();
	~CMEstimatorGPUSparseOPT();

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

	void determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, long* d_tmpIndices, double* d_tmpConfidences, int dim, int kBest);
	culaSparseStatus computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, double* A, int* rowPtr, int* colIdx, double* x, double* b);
	void initIdxDevicePointers(int size, unsigned int dim);
};

#endif /* CMESTIMATORGPUSPARSEOPT_H_ */
