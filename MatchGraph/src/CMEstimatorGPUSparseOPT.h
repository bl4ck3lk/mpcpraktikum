/*
 * CMEstimatorGPUSparseOPT.h
 *
 * Header file for a GPU estimator, implementing the CMEstimator interface.
 * This GPU estimator searches only a certain amount of randomly chosen columns
 * for the k-best values (depending on dimension and requested number of
 * image-pairs).
 *
 *  Created on: Jul 16, 2013
 *      Author: Fabian
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

#endif /* CMESTIMATORGPUSPARSEOPT_H_ */
