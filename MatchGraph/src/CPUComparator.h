/*
 * CPUComparator.h
 *
 * Header file for the comparison module implementing a host version of the
 * image comparison module.
 * The given image-pairs are compared on the host by the image-comparison.
 *
 *  Created on: Jul 13, 2013
 *      Author: Fabian
 */

#ifndef CPUCOMPARATOR_H_
#define CPUCOMPARATOR_H_

#include "ImageComparator.h"
#include "Comparator_CPU.h"

class CPUComparator : public ImageComparator {
public:
	CPUComparator();
	virtual ~CPUComparator();

	//Implemented abstract functions (see ImageComparator.h)
	void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* h_idx1, int* h_idx2, int* h_res, int arraySize);
	void setRandomMode(bool mode);

private:
	//Actually used image comparator
	Comparator_CPU* compCPU;

	//Random mode setting
	bool randomMode;
};

#endif /* CPUCOMPARATOR_H_ */
