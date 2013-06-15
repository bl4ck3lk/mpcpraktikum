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
	GPUMatrix(int dim, float lambda);
	//~GPUMatrix();
	void set(int i, int j, bool val);
	unsigned int getDimension();
	float* getConfMatrixF();
	char* getMatrAsArray();
	char getVal(int i, int j);
	float* getColumn(int i);
	int getSimilarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);
};

#endif /* GPUMATRIX_H_ */
