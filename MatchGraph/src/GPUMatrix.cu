/*
 * GPUMatrix.cpp
 *
 *  Created on: Jun 3, 2013
 *      Author: gufler
 */

#include "GPUMatrix.h"
#include "Tester.h"
#include <cula.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BLOCKS 2
#define MAX_THREADS 8


// Kernel for reducing gridDim.x*blockDim.x (= total num threads) elements to gridDim.x (=num blocks) elements
__global__ void reduceSumKernel(unsigned int *_dst, const char *_srcChar, const unsigned int *_srcInt, const unsigned int _dim)
{
	//shared memory
	extern __shared__ unsigned int partialSum[];

	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

	// index of thread within block
	unsigned int tidx = threadIdx.x;

	// each thread loads one element from global memory to shared memory
	if(i < _dim)
	{
		if(_srcChar != NULL)
			{
				partialSum[tidx] = _srcChar[i] > 0 ? _srcChar[i] : 0;
			}
			else
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
		_dst[blockIdx.x] = partialSum[0];
	}



}


//+++++++++++++++++++++ class stuff ++++++++++++++++++++++++++

GPUMatrix::GPUMatrix()
{
	// TODO Auto-generated constructor stub

}

//GPUMatrix::~GPUMatrix()
//{
//	// TODO Auto-generated destructor stub
//}

void GPUMatrix::init(int _dim, float _lambda)
{
	dim = _dim;
	lambda = _lambda;
	N = dim*dim;
	//FIXME normal malloc
	cudaMallocHost((void**) &data, (N)*sizeof(char));
	memset(data, 0, N);
	cudaMallocHost((void**) &set_idx, (N)*sizeof(unsigned int));
	memset(set_idx, -1, N);
	num_set = 0;
	//FIXME move to constructor?

	//test...
	if(true)
	{
		set(3,7,1); set(7,3, 1);
		set(3,8,1); set(8,3, 1);

		set(1,6, -1); set(6,1, -1);
		set(0,6, -1); set(6,0, -1);
		set(5,6, -1); set(6,5, -1);

		set(0,2,1); set(2,0,1);
		set(4,5,1); set(5,4,1);
		set(2,5,1); set(5,2,1);

		set(1,9,1); set(9,1,1);
	}
}

void GPUMatrix::set(int i, int j, float val)
{
	if(i < dim && j < dim)
	{
		unsigned int index = i * dim + j;
		if(val > 0)
		{
			set_idx[num_set] = index;
			num_set++;
		}
		data[index] = (char)val;
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
	unsigned int degrees[dim]; //FIXME

	// computing the degrees of all nodes (degree matrix for laplacian construction)
	// realized on gpu by computing the sum of -1s and 1s for each row
	char* rowPointer = data;
	for(int row = 0; row < dim; row++, rowPointer += dim)
	{
		dim3 blockGrid(MAX_BLOCKS);
		dim3 threadBlock(MAX_THREADS);

		char* gpuDataRow;
		unsigned int* gpuResult;
		unsigned int* gpuEndResult;
		unsigned int* cpuResult;
		unsigned int* cpuEndResult;

		cudaMallocHost((void**) &cpuResult, MAX_BLOCKS * sizeof(unsigned int));
		cudaMallocHost((void**) &cpuEndResult, sizeof(unsigned int));

		cudaMalloc((void**) &gpuDataRow, dim * sizeof(char));
		cudaMalloc((void**) &gpuResult, MAX_BLOCKS * sizeof(unsigned int));
		cudaMalloc((void**) &gpuEndResult, sizeof(unsigned int));

		//copy row to gpu
		cudaMemcpy(gpuDataRow, rowPointer, dim * sizeof(char), cudaMemcpyHostToDevice);

		reduceSumKernel<<<blockGrid, threadBlock, MAX_THREADS*sizeof(unsigned int)>>>(gpuResult, gpuDataRow, NULL, dim);

		cudaThreadSynchronize();

		bool printItermediate = false;
		if(printItermediate)
		{
			cudaMemcpy(cpuResult, gpuResult, MAX_BLOCKS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			printf("partial: ");
			for (int i = 0; i < MAX_BLOCKS; ++i)
			{
				printf(" %d ", cpuResult[i]);
			}
			printf("\n");
		}


		//need only MAX_BLOCKS number of threads, as there are only MAX_BLOCKS elements left
		dim3 threadBlock2(MAX_BLOCKS);
		//2nd call: reduce to one element, by using only one block
		reduceSumKernel<<<1, threadBlock2, MAX_THREADS*sizeof(int)>>>(gpuEndResult, NULL, gpuResult, dim);

		cudaMemcpy(cpuEndResult, gpuEndResult, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		degrees[row] = cpuEndResult[0];

		printf("degree of %d : %d\n",row,  cpuEndResult[0]);

		//TODO cudaFree
	}

	int laplacian[dim*dim];

	//TODO row-wise on GPU

	for(int i = 0; i < dim; i++)
	{
			for(int j = 0; j < dim; j++)
			{

				char val;
				if(i == j)
				{
					val = 1 + (degrees[i] * lambda * dim);
				}
				else
				{
					int adjacency = getVal(i,j);
					val = (0 - (adjacency > 0 ? adjacency : 0)) * lambda * dim;
				}

				laplacian[i*dim + j] = val;
				if(val < 0)
				{
					printf("  %d ", val);
				}
				else
				{
					printf("   %d ", val);
				}

			}
			printf("\n");
	}

	Tester::testLaplacian(data, laplacian, dim, lambda);

	printf("currently set: %d", num_set);

	//

	return NULL;
} //TODO

char* GPUMatrix::getMatrAsArray()
{
	return NULL;
}//TODO

char GPUMatrix::getVal(int i, int j)
{
	return data[i*dim + j];
}

int GPUMatrix::getSimiliarities()
{
	return -1; //TODO
}

void GPUMatrix::print()
{
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			char val = getVal(i,j);
			if(val < 0)
			{
				printf("%d ", val);
			}
			else
			{
				printf(" %d ", val);
			}

		}
		printf("\n");
	}
}

void GPUMatrix::writeGML(char * filename, bool similar, bool dissimilar, bool potential)
{
	//TODO
}




