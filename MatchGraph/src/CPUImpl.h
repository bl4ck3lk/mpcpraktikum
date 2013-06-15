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

class CPUImpl: public MatrixHandler
{
private:
	Eigen::MatrixXf m;
	int dim;
	float lambda;
	void testInit();
	Eigen::MatrixXf symmetrize(Eigen::MatrixXf F, int dim);
public:
	CPUImpl(int dim, float lambda);
	//~CPUImpl();
	void set(int i, int j, bool val);
	unsigned int getDimension();
	float* getConfMatrixF();
	Eigen::MatrixXf getModLaplacian();
	char* getMatrAsArray();
	char getVal(int i, int j);
	int getSimilarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);
};

#endif /* CPUIMPL_H_ */
