/*
 * CPUImpl.cpp
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#include "CPUImpl.h"
#include <iostream>
#include <fstream>

CPUImpl::CPUImpl()
{
	// TODO Auto-generated constructor stub
}

/*CPUImpl::~CPUImpl()
{
	// TODO Auto-generated destructor stub
}*/

void CPUImpl::init(int _dim, float _lambda)
{
	dim = _dim;
	lambda = _lambda;
	bool test = false;

	if (test) {
		testInit();
	} else {

		//Initialize empty matrix and do random comparisons to fill it
		m = Eigen::MatrixXf::Zero(dim, dim);

		/*
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				m(i, j) = initMatrix[i * dim + j];
			}
		}
		*/

	}
}

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
	writeGML("graphT0.GML", true, true, false);
}

float* CPUImpl::getConfMatrixF()
{
	Eigen::MatrixXf res = Eigen::MatrixXf::Zero(dim, dim);

	//construct W
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


	Eigen::MatrixXf F = Eigen::MatrixXf::Zero(dim, dim);
	for (int j = 0; j < dim; j++) {
		F.col(j) = res.colPivHouseholderQr().solve(m.col(j));
	}

	F = symmetrize(F, dim);

	std::cout << F << std::endl;

	//to float*
	float* arr = new float[dim*dim];
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			arr[i*dim+j] = (float)F(i,j);
		}
	}
	return arr;
}

void CPUImpl::set(int i, int j, float val)
{
	m(i,j) = val;
}

void CPUImpl::print()
{
	std::cout << m << std::endl;
}

unsigned int CPUImpl::getDimension()
{
	return dim;
}

char* CPUImpl::getMatrAsArray()
{
	return NULL; //TODO
}

char CPUImpl::getVal(int i, int j)
{
	return (char)m(i,j);
}

int CPUImpl::getSimiliarities()
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

Eigen::MatrixXf CPUImpl::symmetrize(Eigen::MatrixXf F, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		F(i,i) = 0;
	}
	return (F+F.transpose())/2;
}

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
}
