/*
 * InitializerGPU.h
 *
 * Header file for a GPU initializer, implementing the initializer interface.
 *
 *  Created on: Jun 29, 2013
 *      Author: Fabian
 */

#ifndef INITIALIZERGPU_H_
#define INITIALIZERGPU_H_

#include "Initializer.h"

class InitializerGPU : public Initializer {
public:
	InitializerGPU();
	~InitializerGPU();

	//Implemented abstract function (see Initializer.h)
	void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize);
};

#endif /* INITIALIZERGPU_H_ */
