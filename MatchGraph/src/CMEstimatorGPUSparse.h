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

	int* getIdx1DevicePtr();
	int* getIdx2DevicePtr();
	int* getResDevicePtr();

	int* getIdx1HostPtr(int dim); //todo only for testing purpose
	int* getResHostPtr(int dim); //todo only for testing purpose
	void setResDevicePtr(int* res, int dim); //todo only for testing purpose

	Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest); //not used in this implementation
	void getKBestConfMeasuresSparse(MatrixHandler* T, float* F, int kBest);

	Indices* getInitializationIndices(MatrixHandler* T, int initNr); //not used in this implementation

private:
	int lastSize;

	int* d_idx1;
	int* d_idx2;
	int* d_res;

	int* idx1; //todo only for testing purpose
	int* res; //todo only for testing purpose

	/* cula */

	Indices* getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest);
	int determineBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest, int kBestForThisColumn, int currIndexNr);
	culaSparseStatus computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config, unsigned int dim, unsigned int nnz, float* A, int* rowPtr, int* colIdx, float* x, float* b);
	void initIdxDevicePointers(int size, unsigned int dim);
	void initCula();
};

#endif /* CMESTIMATORGPUSPARSE_H_ */
