/*
 * CPUImpl.h
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#ifndef CPUIMPL_H_
#define CPUIMPL_H_

#include "MatrixHandler.h"
#include "../lib/Eigen/Eigen/Dense"

class CPUImpl : public MatrixHandler{
private:
	Eigen::MatrixXf m;
	int dim;
	float lambda;
	void testInit();
	Eigen::MatrixXf symmetrize(Eigen::MatrixXf F, int dim);
public:
	CPUImpl();
	//~CPUImpl();
	void init(int dim, float lambda);
	void set(int i, int j, float val);
	unsigned int getDimension();
	float* getConfMatrixF();
	Eigen::MatrixXf getModLaplacian();
	char* getMatrAsArray();
	char getVal(int i, int j);
	int getSimiliarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);
	static void testLaplacian(char* gpuData, int* laplacian, int dim, float lambda);
};

#endif /* CPUIMPL_H_ */
