/*
 * Initializer.h
 *
 * Interface for the initializers of the match graph.
 * Before the match graph algorithm starts, initial knowledge about the
 * relations between some of the images is needed. This is gained by random
 * image-comparisons.
 *
 *  Created on: Jul 13, 2013
 *      Author: Fabian
 */

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include "MatrixHandler.h"
#include "ImageComparator.h"
#include "ImageHandler.h"

class Initializer {
public:
	virtual ~Initializer(){};

	//Execute the initializing phase, filling T-Matrix with initial data.
	virtual void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize) = 0;
};

#endif /* INITIALIZER_H_ */
