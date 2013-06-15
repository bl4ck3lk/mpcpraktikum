/*
 * MatrixHandler.h
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#ifndef MATRIXHANDLER_H_
#define MATRIXHANDLER_H_

class MatrixHandler {
public:
	virtual ~MatrixHandler(){};
	virtual void set(int i, int j, bool val) = 0;
	virtual unsigned int getDimension()= 0;
	virtual float* getConfMatrixF()= 0;
	virtual char* getMatrAsArray() = 0;
	virtual char getVal(int i, int j) = 0;
	virtual int getSimilarities() = 0;
	virtual void print() = 0;
	virtual void writeGML(char* filename, bool similar, bool dissimilar, bool potential) = 0;
};

#endif /* MATRIXHANDLER_H_ */
