/*
 * Initializer.h
 *
 *  Created on: Jun 29, 2013
 *      Author: schwarzk
 */

#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include "MatrixHandler.h"
#include "ImageComparator.h"
#include "ImageHandler.h"

class Initializer {
public:
	Initializer(); //constructor
	~Initializer(); //destructor

	/* for T Matrix initialization */
	void doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initAraySize);
};

#endif /* INITIALIZER_H_ */
