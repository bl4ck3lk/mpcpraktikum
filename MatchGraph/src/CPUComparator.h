/*
 * CPUComparator.h
 *
 *  Created on: Jul 13, 2013
 *      Author: schwarzk
 */

#ifndef CPUCOMPARATOR_H_
#define CPUCOMPARATOR_H_

#include "ImageComparator.h"
#include "Comparator_CPU.h"

class CPUComparator : public ImageComparator {
public:
	CPUComparator();
	virtual ~CPUComparator();
	void doComparison(ImageHandler* iHandler, MatrixHandler* T, int* h_idx1, int* h_idx2, int* h_res, int arraySize);
	void setRandomMode(bool mode);

private:
	Comparator_CPU* compCPU;
	bool randomMode;
};

#endif /* CPUCOMPARATOR_H_ */
