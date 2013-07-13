/*
 * GPUComparator.h
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#ifndef GPUCOMPARATOR_H_
#define GPUCOMPARATOR_H_

#include "ImageComparator.h"
#include "Comparator_CVGPU.h"

class GPUComparator : public ImageComparator{
public:
	GPUComparator();
	~GPUComparator();
	//TODO destructor
	void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize);

private:
	ComparatorCVGPU* openCVcomp;

	int* h_idx1;
	int* h_idx2;
	int* h_res;
	int currentArraySize;

	void initArrays(int arraySize);
};

#endif /* GPUCOMPARATOR_H_ */
