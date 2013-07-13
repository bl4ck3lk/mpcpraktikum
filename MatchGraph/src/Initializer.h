/*
 * Initializer.h
 *
 *  Created on: Jul 13, 2013
 *      Author: schwarzk
 */

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include "MatrixHandler.h"
#include "ImageComparator.h"
#include "ImageHandler.h"

class Initializer {
public:
	virtual ~Initializer(){};
	virtual void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize) = 0;
};

#endif /* INITIALIZER_H_ */
