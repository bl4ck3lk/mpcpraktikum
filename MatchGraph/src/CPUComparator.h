/*
 * CPUComparator.h
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#ifndef CPUCOMPARATOR_H_
#define CPUCOMPARATOR_H_

#include "ImageComparator.h"

class CPUComparator : public ImageComparator{
public:
	CPUComparator();
	//TODO destructor
	void doComparison(ImageHandler* iHandler, MatrixHandler* T, int k, Indices* kBestIndices);
	void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize);
};

#endif /* CPUCOMPARATOR_H_ */
