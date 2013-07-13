/*
 * InitializerCPU.h
 *
 *  Created on: Jul 13, 2013
 *      Author: schwarzk
 */

#ifndef INITIALIZERCPU_H_
#define INITIALIZERCPU_H_

#include "Initializer.h"

class InitializerCPU : public Initializer {
public:
	InitializerCPU();
	~InitializerCPU();

	/* for T Matrix initialization */
	void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize);
};

#endif /* INITIALIZERCPU_H_ */
