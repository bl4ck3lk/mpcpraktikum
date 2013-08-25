/*
 * CPUImpl.cpp
 *
 * This class is a matrix representation of the match graph as CPU implementation.
 * For solving the equation system the Eigen library is used. In contrast to the
 * GPU implementation, the equation system is solved in this component and not in
 * the estimator on-the-fly.
 * NOTE: This class is not optimized whatsoever. It is still from the very beginning
 * of this project.
 *
 *  Created on: May 29, 2013
 *      Author: Armin, Fabian
 */

#include "CPUImpl.h"
#include <iostream>
#include <fstream>

/*
 * Constructor
 */
CPUImpl::CPUImpl(int _dim, float _lambda)
{
	dim = _dim;
	lambda = _lambda;
	bool test = true;

	if (test) {
		testInit();
	} else {

		//Initialize empty matrix
		m = Eigen::MatrixXf::Zero(dim, dim);

	}
}

/*
 * Destructor
 */
CPUImpl::~CPUImpl()
{
}

/*
 * Function for debugging and testing.
 * Fills the matrix with highly specific entries.
 */
void CPUImpl::testInit()
{
	m = Eigen::MatrixXf::Zero(dim, dim);

	m(3,7) = 1; m(7,3) = 1;
	m(3,8) = 1; m(8,3) = 1;

	m(1,6) = -1; m(6,1) = -1;
	m(0,6) = -1; m(6,0) = -1;
	m(5,6) = -1; m(6,5) = -1;

	m(0,2) = 1; m(2,0) = 1;
	m(4,5) = 1; m(5,4) = 1;
	m(2,5) = 1; m(5,2) = 1;

	m(1,9) = 1; m(9,1) = 1;
	//writeGML("graphT0.GML", true, true, false);
}

/*
 * Returns the confidence measure matrix for the current match graph.
 * Solves implicitly the equation systems using Eigen library with a
 * householder rank-revealing QR decomposition solver.
 */
float* CPUImpl::getConfMatrixF()
{
	Eigen::MatrixXf laplacian = getModLaplacian();

	Eigen::MatrixXf F = Eigen::MatrixXf::Zero(dim, dim);
	for (int j = 0; j < dim; j++) {
		F.col(j) = laplacian.colPivHouseholderQr().solve(m.col(j));
	}

	F = symmetrize(F, dim);

	//std::cout << "F Matrix is: \n" << F << std::endl;

	//to float*
	float* arr = new float[dim*dim];
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			arr[i*dim+j] = (float)F(i,j);
		}
	}

	printf("No CPU solver measurement done.\t");

	return arr;
}

/*
 * Returns the modified laplacian for the current match graph.
 */
Eigen::MatrixXf CPUImpl::getModLaplacian()
{
	Eigen::MatrixXf res = Eigen::MatrixXf::Zero(dim, dim);

	//construct W //TODO do not need to explicitly construct W
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (m(i, j) > 0) {
				res(i, j) = m(i, j);
			}
		}
	}

	//TODO consider using Degree-Matrix D and Adjacency-Matrix A to compute laplacian as D - A.
	//construct laplacian (modified with *lambda*res+Identity
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			if (i == j) {
				int neighbours = 0;
				for (int k = 0; k < dim; k++) {
					res(i, k) ? neighbours++ : neighbours;
				}
				res(i, j) = neighbours * lambda * dim + 1;
			} else {
				res(i, j) = (res(i, j) == 0) ? 0 : -res(i, j);
				res(i, j) *= (lambda * dim);
			}
		}
	}

	//std::cout << "Mod Laplacian is: \n" << res << std::endl;

	return res;
}

/*
 * Set a specific entry of the matrix to true or false.
 * (Symmetric entry is not set implicitly).
 */
void CPUImpl::set(int i, int j, bool val)
{
	m(i,j) = val ? 1 : -1;
}

/*
 * Sets all given entries to the given results.
 * (Symmetric entries are set implicitly).
 */
void CPUImpl::set(int* idx1, int* idx2, int* res, int size)
{
	for(int i = 0; i < size; i++)
	{
		int x = idx1[i];
		int y = idx2[i];
		m(x,y) = (res[i] == 1) ? 1 : -1;
		m(y,x) = (res[i] == 1) ? 1 : -1;
	}
}

/*
 * Print the current match graph matrix on the console.
 */
void CPUImpl::print()
{
	std::cout << m << std::endl;
}

/*
 * Return the dimension of the matrix.
 */
unsigned int CPUImpl::getDimension()
{
	return dim;
}

/*
 * Return the matrix as array. (Not implemented).
 */
char* CPUImpl::getMatrAsArray()
{
	return NULL; //TODO
}

/*
 * Return the current value of a specific matrix entry.
 */
char CPUImpl::getVal(int i, int j)
{
	return (char)m(i,j);
}

/*
 * Return the number of image-pairs that are marked as 'similar'.
 */
int CPUImpl::getSimilarities()
{
	int similarities = 0;
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			if (m(i,j) == 1) similarities++;
		}
	}
	return similarities;
}

/*
 * Return a symmetrized matrix of the given confidence measure matrix.
 */
Eigen::MatrixXf CPUImpl::symmetrize(Eigen::MatrixXf F, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		F(i,i) = 0;
	}
	return (F+F.transpose())/2;
}

/*
 * Write the current match graph matrix as GraphML file.
 */
void CPUImpl::writeGML(char* filename, bool similar, bool dissimilar, bool potential){
	bool show_potential = potential;
	bool show_similar = similar;
	bool show_dissimilar = dissimilar;

	std::ofstream file;
	file.open(filename);
	file << "graph\n[\nhierarchic 1\ndirected 0\n";
	for(int i = 0; i < dim; i++){
		file << "node\n\t[ id "<< i << "\n\tgraphics\n\t[ type \"circle\"]"<<
				"\n\tLabelGraphics\n\t[text \""<<i<<"\"]\n]\n";
	}
	for(int i = 0; i < dim; i++)
		{
			for(int j = i+1; j < dim; j++)
			{
				if(i==j)
					continue;

				std::string style;
				std::string fill;
				if(m(i,j) > 0){
					if(!show_similar)
						continue;
					style = "line";
					fill = "FFFFFF";
				}
				else if(m(i,j) < 0){
					if(!show_dissimilar)
						continue;
					style = "line";
					fill = "#FF0000";
				}
				else{
					if(!show_potential)
						continue;
					style = "dashed";
					fill = "FFFFFF";
				}

				file << "edge\n\t[ source "<< i <<"\n\t  target "<< j<< "\n"<<
					"\tgraphics [style \""<< style <<"\" fill \""<<fill <<"\"]\n]\n";
			}
		}
	file << "]\n";

	file.close();
}


