/*
 * GPUSparse.h
 *
 *  Created on: Jun 12, 2013
 *      Author: gufler
 */

#ifndef GPUSPARSE_H_
#define GPUSPARSE_H_

#include "MatrixHandler.h"
#include <boost/dynamic_bitset.hpp>

class GPUSparse : public MatrixHandler{
private:
	unsigned int dim;
	unsigned long N;
	float lambda;
	unsigned long num_set;
	char* data; // upper half of T matrix (without diagonal)
				// has dim(dim-1) / 2 elements

public:
	GPUSparse();
	GPUSparse(unsigned int _dim, float _lambda);
	//~GPUMatrix();
	void set(int i, int j, bool val);
	void set(int idx, bool val);
	unsigned int getDimension();
	float* getConfMatrixF();
	char* getMatrAsArray();
	char getVal(int i, int j);
	float* getColumn(int i);
	int getSimilarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);
};

#endif /* GPUSPARSE_H_ */
