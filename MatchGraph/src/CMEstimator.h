/*
 * CMEstimator.h
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#ifndef CMESTIMATOR_H_
#define CMESTIMATOR_H_

#include "MatrixHandler.h"

struct Indices {
	int i, j;
	
	//constructor initializes to -1
	Indices() : i(-1), j(-1) {}
};

struct Entry {
    float value;
    int i, j;
};

class CMEstimator {
public:
	virtual Indices* getKBestConfMeasures(MatrixHandler* T, float* F, int kBest) = 0;
	virtual Indices* getInitializationIndices(MatrixHandler* T, int initNr) = 0;
	virtual ~CMEstimator(){};
};

#endif /* CMESTIMATOR_H_ */
