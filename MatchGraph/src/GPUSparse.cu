/*
 * GPUSparse.cpp
 *
 *  Created on: Jun 12, 2013
 *      Author: gufler
 */

#include "GPUSparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

GPUSparse::GPUSparse()
{
	// TODO Auto-generated constructor stub
}

GPUSparse::GPUSparse(unsigned int _dim, float _lambda)
	: dim(_dim), lambda(_lambda), N((_dim*(_dim-1))/2)
{
	data = (char*) malloc(N*sizeof(char));
	memset(data, 0, N);
	num_set = 0;
}


void GPUSparse::set(int i, int j, bool val)
{
	if (i == j) return;
	if (i > j)
	{
		int tmp = i;
		i = j;
		j = tmp;
	}
	// map to upper 'diagonal' without the actual diagonal
	int idx = (i*dim + j) - ((i+1)*(i+2)*.5);
	data[idx] = val ? 1 : -1;
}

void GPUSparse::set(int idx, bool val){
	data[idx] = val ? 1 : -1;
}

unsigned int GPUSparse::getDimension()
{
	return dim;
}

float* GPUSparse::getConfMatrixF()
{
	//SPARSE CSC format for modified LAPLACIAN
	int nnz = num_set; //number of non-zero elements
	float* vals; //array with the values
	vals = (float*) malloc (nnz * sizeof(float));
	int* rowIdx; //row index
	rowIdx = (int*) malloc (nnz * sizeof(int));
	int* colPtr; //column pointer (dim+1) elements, last entry points to one past final data element.
	colPtr = (int*) malloc ((dim+1) * sizeof(int));


//	int laplacian[dim*dim];
//	float _cpuLambda_times_dim = dim * lambda;
//	cudaMemcpyToSymbol(lambda_times_dim, &_cpuLambda_times_dim, sizeof(float));
//
//	const int MAX_THREADS = 128;
//	const int NUM_BLOCKS = (dim + MAX_THREADS - 1) / MAX_THREADS;
//	dim3 blockGrid(NUM_BLOCKS);
//	dim3 threadBlock(MAX_THREADS);
//	dim3 threadBlock2(NUM_BLOCKS);
//
//	char* gpuDataRow;
//	int* gpuSetIndices;
//	unsigned int* gpuResult;
//	unsigned int* gpuEndResult;
//	unsigned int* cpuResult;
//	unsigned int* cpuEndResult;
//
//	cudaMallocHost((void**) &cpuResult, NUM_BLOCKS * sizeof(unsigned int));
//	cudaMallocHost((void**) &cpuEndResult, sizeof(unsigned int));
//
//	cudaMalloc((void**) &gpuSetIndices, dim * sizeof(int));
//	cudaMalloc((void**) &gpuDataRow, dim * sizeof(char));
//	cudaMalloc((void**) &gpuResult, NUM_BLOCKS * sizeof(unsigned int));
//	cudaMalloc((void**) &gpuEndResult, sizeof(unsigned int));
//
//	// computing the degrees of all nodes (degree matrix for laplacian construction)
//	// realized on gpu by computing the sum of -1s and 1s for each row
//	char* rowPointer = data;
//	int* laplacianRowPointer = laplacian;
//	for(int row = 0; row < dim; row++, rowPointer += dim, laplacianRowPointer += dim)
//	{
//
//		//copy row to gpu
//		cudaMemcpy(gpuDataRow, rowPointer, dim * sizeof(char), cudaMemcpyHostToDevice);
//
//		//reduceSumKernel<<<blockGrid, threadBlock, MAX_THREADS*sizeof(unsigned int)>>>(gpuResult, gpuSetIndices, gpuDataRow, NULL, dim);
//
//		cudaThreadSynchronize();
//
//		bool printItermediate = false;
//		if(printItermediate)
//		{
//			cudaMemcpy(cpuResult, gpuResult, NUM_BLOCKS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//			printf("partial: ");
//			Tester::printArrayInt((int*)cpuResult, NUM_BLOCKS);
//		}
//
//		//2nd call: reduce to one element, by using only one block
//		//reduceSumKernel<<<1, threadBlock2, MAX_THREADS*sizeof(int)>>>(gpuEndResult, NULL, NULL, gpuResult, dim);
//
//		//Download results from GPU
//		cudaMemcpy(laplacianRowPointer, gpuSetIndices, dim*sizeof(int), cudaMemcpyDeviceToHost);
//		cudaMemcpy(cpuEndResult, gpuEndResult, sizeof(unsigned int), cudaMemcpyDeviceToHost);
//
//		//set the diagonal element
//		laplacianRowPointer[row] = cpuEndResult[0];
//
//		if(printItermediate)
//		{
//			printf("row %d : ", row);
//			Tester::printArrayInt(laplacianRowPointer, dim);
//			printf("degree of %d : %d\n",row,  cpuEndResult[0]);
//		}
//
//		//TODO cudaFree
//	}
//
//	//Tester::printMatrixArrayInt(laplacian, dim);
//	Tester::testLaplacian(data, laplacian, dim, lambda);
//
//	getColumn(3);
//
//	printf(" used %d THREADS and %d BLOCKS.", MAX_THREADS, NUM_BLOCKS );
//
//	printf("\n currently set elems: %d \n", num_set);
//
//
//	return NULL;
} //TODO

float* GPUSparse::getColumn(int i)
{
 //TODO
	return NULL;
}

char* GPUSparse::getMatrAsArray()
{
	return NULL;
}//TODO

char GPUSparse::getVal(int i, int j)
{
	return 'c';
}

int GPUSparse::getSimilarities()
{
	return 0;
}

void GPUSparse::print()
{
	//TODO
}

void GPUSparse::writeGML(char * filename, bool similar, bool dissimilar, bool potential)
{
	//TODO
}
