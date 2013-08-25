/*
 * GPUComparator.h
 *
 * Header file for the comparison module implementing a device version of the
 * image comparison module.
 * The given image-pairs are compared on device by the image-comparison.
 *
 *  Created on: May 29, 2013
 *      Author: Armin, Fabian
 */

#ifndef GPUCOMPARATOR_H_
#define GPUCOMPARATOR_H_

#include "ImageComparator.h"
#include "Comparator_CVGPU.h"

class GPUComparator : public ImageComparator{
public:
	GPUComparator();
	~GPUComparator();

	//Implemented abstract functions (see ImageComparator.h)
	void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize);
	void setRandomMode(bool mode);

private:
	//Actually used image comparator
	ComparatorCVGPU* openCVcomp;

	//Random mode setting
	bool randomMode;

	//Host buffers for the image comparison
	int* h_idx1;
	int* h_idx2;
	int* h_res;

	//Current size of buffer arrays
	int currentArraySize;

	//Allocate buffer arrays on host
	void initArrays(int arraySize);

	//Runtime storage
	__int64_t totalTime;
};

#endif /* GPUCOMPARATOR_H_ */
