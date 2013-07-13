/*
 * InitializerGPU.h
 *
 *  Created on: Jun 29, 2013
 *      Author: schwarzk
 */

#ifndef INITIALIZERGPU_H_
#define INITIALIZERGPU_H_

#include "Initializer.h"

class InitializerGPU : public Initializer {
public:
	InitializerGPU();
	~InitializerGPU();

	/* for T Matrix initialization */
	void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize);
};

#endif /* INITIALIZERGPU_H_ */
