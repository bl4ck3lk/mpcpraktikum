/*
 * CMEstimatorGPUSparseMax.h
 *
 * Header file for a GPU estimator, implementing the CMEstimator interface.
 * This GPU estimator searches the overall k-best values.
 *
 *  Created on: Jul 16, 2013
 *      Author: Fabian
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

	//Implemented abstract functions (see CMEstimator.h)
	void getKBestConfMeasures(MatrixHandler* T, float* F, int kBest);
	void computeRandomComparisons(MatrixHandler* T, const int k);
	int* getIdx1Ptr();
	int* getIdx2Ptr();
	int* getResPtr();

	//Functions for debugging and testing
	int* getIdx1HostPtr(int dim);
	int* getResHostPtr(int dim);
	void setResDevicePtr(int* res, int dim);


private:
	//Runtime storage
	__int64_t totalTime;
	
	//Device memory for resulting arrays managed by the specific estimator
	int* d_idx1;
	int* d_idx2;
	int* d_res;

	//Current size of resulting arrays
	int lastSize;

	//Host memory for using the debug and testing functions (otherwise unused)
	int* idx1;
	int* res;

	//Allocate resulting arrays on device
	void initIdxDevicePointers(int size, unsigned int dim);

	//Functions to solve for the confidence measure matrix (with CULA)
	void determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, long* d_tmpIndices, double* d_tmpConfidences, int dim, int kBest);
	culaSparseStatus computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, double* A, int* rowPtr, int* colIdx, double* x, double* b);
};

#endif /* CMESTIMATORGPUSPARSEMAX_H_ */
