/*
 * GPUMatrix.h
 *
 *  Created on: Jun 3, 2013
 *      Author: gufler
 */

#ifndef GPUMATRIX_H_
#define GPUMATRIX_H_

#include "MatrixHandler.h"

class GPUMatrix : public MatrixHandler{
private:
	char* data; //T matrix
	unsigned int* set_idx;
	unsigned int num_set;
	int dim;
	unsigned int N;
	float lambda;
	void setOnHost(int, int, char);
public:
	GPUMatrix();
	//~GPUMatrix();
	void init(int dim, float lambda);
	void set(int i, int j, float val);
	unsigned int getDimension();
	float* getConfMatrixF();
	char* getMatrAsArray();
	char getVal(int i, int j);
	int getSimiliarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);
};

#endif /* GPUMATRIX_H_ */
