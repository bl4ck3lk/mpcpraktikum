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
};

#endif /* CPUCOMPARATOR_H_ */
