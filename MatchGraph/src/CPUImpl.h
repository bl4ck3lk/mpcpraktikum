/*
 * CPUImpl.h
 *
 * CPU implementation of the MatrixHandler interface.
 * This class is a match graph matrix representation for the algorithm,
 * providing all the necessary functions for execution.
 *
 *  Created on: May 29, 2013
 *      Author: Armin, Fabian
 */

#ifndef CPUIMPL_H_
#define CPUIMPL_H_

#include "MatrixHandler.h"
#include "../lib/Eigen/Eigen/Dense"

class CPUImpl: public MatrixHandler
{
private:
	//Storage for the matrix representation of the match graph.
	Eigen::MatrixXf m;

	//Dimension of the matrix.
	int dim;

	//Used lambda value.
	float lambda;

	//Symmetrize the given confidence measure matrix.
	Eigen::MatrixXf symmetrize(Eigen::MatrixXf F, int dim);

	//For testing and debugging. Fill the matrix with specific dummy data.
	void testInit();

public:
	CPUImpl(int dim, float lambda);
	~CPUImpl();

	//Implemented abstract functions (see MatrixHandler.h)
	void set(int i, int j, bool val);
	unsigned int getDimension();
	float* getConfMatrixF();
	char* getMatrAsArray();
	char getVal(int i, int j);
	int getSimilarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);

	//Update the T-Matrix with an array of new results.
	void set(int* idx1, int* idx2, int* res, int size);

	//Return the modified laplacian of the T-Matrix
	Eigen::MatrixXf getModLaplacian();
};

#endif /* CPUIMPL_H_ */
