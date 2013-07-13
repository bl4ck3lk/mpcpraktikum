/*
 * CMEstimatorGPUApprox.cpp
 *
 * Generates a list of indices containing the i, j index of approx. the 
 * k-best confidence measure values. 
 * This implementation handles arbitrary large matrices, as long as one
 * row of the confidence measure matrix and one int array of the same
 * size fits into device memory.
 *
 *  Created on: 05.06.2013
 *      Author: Fabian
 */

#include "CMEstimatorGPUApprox.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <algorithm> /* std::find */
#include <stdio.h> /* printf */
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define CUDA_CHECK_ERROR() {							\
    cudaError_t err = cudaGetLastError();					\
    if (cudaSuccess != err) {						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
}

const int THREADS = 256;
//const unsigned long MAX_DEVICE_MEMORY_BYTES = 512; //for testing purpose with dim=10
const unsigned long MAX_DEVICE_MEMORY_BYTES = 1073741824; //1 GB

//Initialize indices
static __global__ void initIndicesKernel(int* gpuIndices, const int elements, const int offset, const int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < elements)
	{
		/* :TRICKY:
		 * Any index in the lower diagonal matrix incl. diagonal elements
	 	 * will contain a -1, otherwise the index nr.
		 */
		if ((idx+offset) < 1+((idx+offset)/dim)+((idx+offset)/dim)*dim)
		{
			gpuIndices[idx] = -1;
		}
		else
		{
			gpuIndices[idx] = idx+offset;
		}
	}
}

CMEstimatorGPUApprox::CMEstimatorGPUApprox() {
	// TODO Auto-generated constructor stub

}

Indices* CMEstimatorGPUApprox::getInitializationIndices(MatrixHandler* T, int initNr)
{
	Indices* initIndices = new Indices[initNr];
	std::vector<int> chosenOnes; //max size will be initNr
	int dim = T->getDimension();

	//generate random index
	srand (time(NULL));
	const int MAX_ITERATIONS = dim*(dim/2) + dim; //#elements in upper diagonal matrix + dim

	//generate initialization indices
	for(int i = 0; i < initNr; i++)
	{
		int rIdx = -1;
		int x, y;

		int c = 0;
		do {
			//get random number
			rIdx = rand() % (dim*dim);

			//compute matrix indices with given continuous index sequence
			x = rIdx/dim;
			y = rIdx%dim;
			c++;
		} while ( ((rIdx < 1+(rIdx/dim)+(rIdx/dim)*dim)
					|| (T->getVal(x,y) != 0)
					|| (std::find(chosenOnes.begin(), chosenOnes.end(), rIdx) != chosenOnes.end()))
				&& (c <= MAX_ITERATIONS) );
		/* :TRICKY:
		 * As long as the random number is not within the upper diagonal matrix w/o diagonal elements
		 * or T(idx) != 0 generate or already in the list of Indices, a new random index but maximal
		 * MAX_ITERAtION times.
		 */

		if (c <= MAX_ITERATIONS) //otherwise initIndices contains -1 per struct definition
		{
			chosenOnes.push_back(rIdx);
			initIndices[i].i = x;
			initIndices[i].j = y;
		}
	}

	return initIndices;
}

Indices* CMEstimatorGPUApprox::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("Determine kBest confidence measures (approx.) on GPU:\n");
	int dim = T->getDimension();

	//storage for the kBest indices
	Indices* kBestIndices = new Indices[kBest];
	int countBest = 0;

	//Two arrays have to fit in device memory (min. size = dim = one row)
	unsigned long nrRowsFitInMemory = MAX_DEVICE_MEMORY_BYTES / (dim*sizeof(float)+dim*sizeof(int));
	if (nrRowsFitInMemory > dim)
	{
		nrRowsFitInMemory = dim; //whole matrix fits in memory
	}
	unsigned int nrCyclesForMatrix = (dim%nrRowsFitInMemory == 0) ? (dim/nrRowsFitInMemory) : (dim/nrRowsFitInMemory) + 1;
	unsigned int elements = nrRowsFitInMemory*dim; //per Cycle

	//debug
	if (true)
	{
		printf("\tCMEstimatorGPUApprox:\n");
		printf("\tNumber of cycles for whole matrix: %i\n",nrCyclesForMatrix);
		printf("\tMax. number of rows per cycle: %lu\n",nrRowsFitInMemory);
		printf("\tNumber of elements processed per cycle: %i\n", elements);
	}

	//Allocate arrays on device
	int* gpuIndices;
	cudaMalloc((void**)&gpuIndices, elements*sizeof(int));
	CUDA_CHECK_ERROR();
	//wrap raw pointer with device pointer
	thrust::device_ptr<int> dp_gpuIndices = thrust::device_pointer_cast(gpuIndices);
	CUDA_CHECK_ERROR();

	//Kernel settings
	int numBlocks = (elements + THREADS - 1) / THREADS;
	dim3 threadBlock(THREADS);
	dim3 blockGrid(numBlocks);

	//Process matrix as long as indices are needed or till done
	for(int i = 0; i < nrCyclesForMatrix && countBest < kBest; i++)
	{
		//init indices array such that indices = [-1,1,2,...], whereas the diagonal elements
		//and the lower diagonal matrix contains -1. These entries should not be chosen
		//later because the matrix is symmetric.
		unsigned int offset = i*elements;
		initIndicesKernel<<<blockGrid, threadBlock>>>(gpuIndices, elements, offset, dim);
		CUDA_CHECK_ERROR();

		//copy part of confidence measure matrix to device
		unsigned int upTo = (offset+elements > dim*dim) ? dim*dim : offset+elements-1;
		thrust::device_vector<float> dev_F(F + offset, F + upTo);
		CUDA_CHECK_ERROR();

		//sort CM-matrix on GPU and sort indices array respectively
		thrust::sort_by_key(dev_F.begin(), dev_F.end(), dp_gpuIndices, thrust::greater<float>());
		CUDA_CHECK_ERROR();

		//download device memory
		int* indices = new int[elements];
		thrust::copy(dp_gpuIndices, dp_gpuIndices+elements, indices);
		CUDA_CHECK_ERROR();

		//get some best indices for these rows
		int chosen = 0;
		for(int j = 0; j < elements && countBest < kBest; j++)
		{
			int idx = indices[j];
			int x = idx/dim;
			int y = idx%dim;

			if (-1 != idx && 0 == T->getVal(x,y)) //only upper diagonal matrix, where T is zero
			{
				kBestIndices[countBest].i = x;
				kBestIndices[countBest].j = y;
				countBest++;
				chosen++;
			}
			//unused kBest-slots contain "-1" as indices per struct definition

			/* choose (kBest/(nrCyclesForMatrix-1)) indices per turn if possible,
			 * but if there aren't any left (-1 or T(x,y)=0) add the not chosen slots
			 * to the next turn to avoid getting much less than desired.
			 */
			if ((1 < nrCyclesForMatrix) && (chosen >= (kBest/(nrCyclesForMatrix-1)) + (i*(kBest/(nrCyclesForMatrix-1)) - countBest)))
				break;
		}
	}

	//free gpu memory
	cudaFree(gpuIndices);
	//dev_F is automatically freed by thrust

	//print
	if (true)
	{
		printf("%i best entries:\n", kBest);
		for(int i = 0; i < kBest; i++)
		{
			if (kBestIndices[i].i != -1)
			{
				//value can't be printed because it is not saved in the Indices-list
				printf("%i: at [%i,%i]\n",i,kBestIndices[i].i,kBestIndices[i].j);
			}
		}
	}

	return kBestIndices;
}

