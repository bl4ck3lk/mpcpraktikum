/*
 * InitializerCPU.h
 *
 * Header file for a CPU initializer, implementing the initializer interface.
 *
 *  Created on: Jul 13, 2013
 *      Author: Fabian
 */

#ifndef INITIALIZERCPU_H_
#define INITIALIZERCPU_H_

#include "Initializer.h"

class InitializerCPU : public Initializer {
public:
	InitializerCPU();
	~InitializerCPU();

	//Implemented abstract function (see Initializer.h)
	void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize);
};

#endif /* INITIALIZERCPU_H_ */
