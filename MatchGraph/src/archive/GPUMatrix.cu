/*
 * GPUMatrix.cpp
 *
 *  Created on: Jun 3, 2013
 *      Author: gufler
 */

#include "GPUMatrix.h"
#include "Tester.h"
#include "cula.h"
#include <stdio.h>
#include <stdlib.h>

__device__ __constant__ float lambda_times_dim;

//FIXME not correct when row longer than number of threads
//TODO split into two kernels or more

// Kernel for reducing gridDim.x*blockDim.x (= total num threads) elements to gridDim.x (=num blocks) elements
__global__ void reduceSumKernel(unsigned int *_dst, int *_setIndices,  const char *_srcChar, const unsigned int *_srcInt, const unsigned int _dim)
{
	//shared memory
	extern __shared__ unsigned int partialSum[];

	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

	// index of thread within block
	unsigned int tidx = threadIdx.x;

	// each thread loads one element from global memory to shared memory
	if (i < _dim)
	{
		if (_srcChar != NULL)
		{ //first mode, where we reduce to MAX_BLOCK elements and have char* input.

			char srcVal = _srcChar[i];
			if(srcVal > 0)
			{
				partialSum[tidx] = srcVal;
				_setIndices[i] = -lambda_times_dim;
			}
			else
			{
				partialSum[tidx] = 0;
				_setIndices[i] = 0;
			}
		}
		else // we are in the second mode (reduce block sums to one element) and have int* input.
		{
			partialSum[tidx] = _srcInt[i];
		}
	}


	__syncthreads();


//faster version (according to slide no 28)
	for(unsigned int stride = blockDim.x/2; stride >= 1; stride >>= 1)
	{
		if(tidx < stride)
		{
			const char val = partialSum[tidx+stride];
			if(val > 0)
			{
				partialSum[tidx] += val;
			}
		}
		__syncthreads();
	}


	// thread with index 0 writes result
	if(tidx == 0)
	{
		//TODO *lambda*dim+1
		if(_srcChar == NULL)
			_dst[blockIdx.x] = partialSum[0] * lambda_times_dim + 1;
		else
			_dst[blockIdx.x] = partialSum[0];
	}

}


//+++++++++++++++++++++ class stuff ++++++++++++++++++++++++++

GPUMatrix::GPUMatrix(int _dim, float _lambda)
{
	dim = _dim;
	lambda = _lambda;
	N = dim * dim;
	data = (char*) malloc(N * sizeof(char));
	memset(data, 0, N);
	set_idx = (unsigned int*) malloc(N * sizeof(unsigned int));
	memset(set_idx, -1, N);
	num_set = 0;
	//TODO move to constructor?

	//test...
	if (true)
	{
		set(3, 7, 1);
		set(7, 3, 1);
		set(3, 8, 1);
		set(8, 3, 1);

		set(1, 6, -1);
		set(6, 1, -1);
		set(0, 6, -1);
		set(6, 0, -1);
		set(5, 6, -1);
		set(6, 5, -1);

		set(0, 2, 1);
		set(2, 0, 1);
		set(4, 5, 1);
		set(5, 4, 1);
		set(2, 5, 1);
		set(5, 2, 1);

		set(1, 9, 1);
		set(9, 1, 1);
	}
}

//GPUMatrix::~GPUMatrix()
//{
//	// TODO Auto-generated destructor stub
//}

void GPUMatrix::set(int i, int j,  bool val)
{
	if(i < dim && j < dim)
	{
		unsigned int index = i * dim + j;
		if(val > 0)
		{
			set_idx[num_set] = index;
			num_set++;
		}
		data[index] = val ? 1 : -1;
	}
	else
	{
		//TODO error handling
	}

}

unsigned int GPUMatrix::getDimension()
{
	return dim;
}

float* GPUMatrix::getConfMatrixF()
{
	//SPARSE CSC format for modified LAPLACIAN
	int nnz = num_set; //number of non-zero elements
	float* vals; //array with the values
	vals = (float*) malloc (nnz * sizeof(float));
	int* rowIdx; //row index
	rowIdx = (int*) malloc (nnz * sizeof(int));
	int* colPtr; //column pointer (dim+1) elements, last entry points to one past final data element.
	colPtr = (int*) malloc ((dim+1) * sizeof(int));



	int laplacian[dim*dim];
	float _cpuLambda_times_dim = dim * lambda;
	cudaMemcpyToSymbol(lambda_times_dim, &_cpuLambda_times_dim, sizeof(float));

	const int MAX_THREADS = 128;
	const int NUM_BLOCKS = (dim + MAX_THREADS - 1) / MAX_THREADS;
	dim3 blockGrid(NUM_BLOCKS);
	dim3 threadBlock(MAX_THREADS);
	dim3 threadBlock2(NUM_BLOCKS);

	char* gpuDataRow;
	int* gpuSetIndices;
	unsigned int* gpuResult;
	unsigned int* gpuEndResult;
	unsigned int* cpuResult;
	unsigned int* cpuEndResult;

	cudaMallocHost((void**) &cpuResult, NUM_BLOCKS * sizeof(unsigned int));
	cudaMallocHost((void**) &cpuEndResult, sizeof(unsigned int));

	cudaMalloc((void**) &gpuSetIndices, dim * sizeof(int));
	cudaMalloc((void**) &gpuDataRow, dim * sizeof(char));
	cudaMalloc((void**) &gpuResult, NUM_BLOCKS * sizeof(unsigned int));
	cudaMalloc((void**) &gpuEndResult, sizeof(unsigned int));

	// computing the degrees of all nodes (degree matrix for laplacian construction)
	// realized on gpu by computing the sum of -1s and 1s for each row
	char* rowPointer = data;
	int* laplacianRowPointer = laplacian;
	for(int row = 0; row < dim; row++, rowPointer += dim, laplacianRowPointer += dim)
	{

		//copy row to gpu
		cudaMemcpy(gpuDataRow, rowPointer, dim * sizeof(char), cudaMemcpyHostToDevice);

		reduceSumKernel<<<blockGrid, threadBlock, MAX_THREADS*sizeof(unsigned int)>>>(gpuResult, gpuSetIndices, gpuDataRow, NULL, dim);

		cudaThreadSynchronize();

		bool printItermediate = false;
		if(printItermediate)
		{
			cudaMemcpy(cpuResult, gpuResult, NUM_BLOCKS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			printf("partial: ");
			Tester::printArrayInt((int*)cpuResult, NUM_BLOCKS);
		}

		//2nd call: reduce to one element, by using only one block
		reduceSumKernel<<<1, threadBlock2, MAX_THREADS*sizeof(int)>>>(gpuEndResult, NULL, NULL, gpuResult, dim);

		//Download results from GPU
		cudaMemcpy(laplacianRowPointer, gpuSetIndices, dim*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuEndResult, gpuEndResult, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		//set the diagonal element
		laplacianRowPointer[row] = cpuEndResult[0];

		if(printItermediate)
		{
			printf("row %d : ", row);
			Tester::printArrayInt(laplacianRowPointer, dim);
			printf("degree of %d : %d\n",row,  cpuEndResult[0]);
		}

		//TODO cudaFree
	}

	//Tester::printMatrixArrayInt(laplacian, dim);
	Tester::testLaplacian(data, laplacian, dim, lambda);

	getColumn(3);

	printf(" used %d THREADS and %d BLOCKS.", MAX_THREADS, NUM_BLOCKS );

	printf("\n currently set elems: %d \n", num_set);

	// ==== SOLVING =====
//	culaStatus s;
//
//	s = culaInitialize();
//	if(s != culaNoError)
//	{
//	    printf("%s dd \n", culaGetStatusString(s));
//	    /* ... Error Handling ... */
//	}
//
//	/* ... Your code ... */
//
//	culaShutdown();


	return NULL;
} //TODO

float* GPUMatrix::getColumn(int i)
{
	float* col;
	col = (float*) malloc(dim*sizeof(float));

	int idx = i;
	for(int j = 0 ; j < dim; j++, idx+=dim)
	{
		col[j] = data[idx];
	}

	Tester::printArrayFloat(col, dim);
	return col;
}

char* GPUMatrix::getMatrAsArray()
{
	return NULL;
}//TODO

char GPUMatrix::getVal(int i, int j)
{
	return data[i*dim + j];
}

int GPUMatrix::getSimilarities()
{
	return num_set;
}

void GPUMatrix::print()
{
	Tester::printMatrixArrayChar(data, dim);
}

void GPUMatrix::writeGML(char * filename, bool similar, bool dissimilar, bool potential)
{
	//TODO
}




