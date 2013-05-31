/*
 * CMEstimatorGPUSorted.cpp
 *
 *  Created on: 30.05.2013
 *      Author: Fabian
 *
 * This class finds the kBest values in a given array on GPU.
 */

#include "CMEstimatorGPUSorted.h"
#include <thrust/sort.h>

#define CUDA_CHECK_ERROR() {							\
    cudaError_t err = cudaGetLastError();					\
    if (cudaSuccess != err) {						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
}

const int THREADS = 256;

//Initialize indices
static __global__ void initIndicesKernel(int* gpuIndices, const int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < dim*dim)
	{
		/* :TRICKY:
		 * Any index in the lower diagonal matrix incl. diagonal elements
	 	 * will contain a -1, otherwise the index nr.	 
		 */
		if (idx < 1+(idx/dim)+(idx/dim)*dim)
		{
			gpuIndices[idx] = -1;
		}
		else
		{
			gpuIndices[idx] = idx;
		}
	}
}

CMEstimatorGPUSorted::CMEstimatorGPUSorted() {
	// TODO Auto-generated constructor stub

}

//TODO use gpu for this (just copied from CMEEstimatorCPU)
Indices* CMEstimatorGPUSorted::getInitializationIndices(MatrixHandler* T, int initNr)
{
	Indices* initIndices = new Indices[initNr];
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
		} while ( ((rIdx > 1+(rIdx/dim)+(rIdx/dim)*dim) || (T->getVal(x,y) != 0))
				&& (c <= MAX_ITERATIONS) );
		/* :TRICKY:
		 * As long as the random number is not within the upper diagonal matrix w/o diagonal elements
		 * or T(idx) != 0 generate a new random index but maximal MAX_ITERAtION times.
		 */

		if (c <= MAX_ITERATIONS) //otherwise initIndices contains -1 per struct definition
		{
			initIndices[i].i = x;
			initIndices[i].j = y;
		}
	}

	return initIndices;
}

Indices* CMEstimatorGPUSorted::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("Determine kBest confidence measures on GPU:\n");
	int dim = T->getDimension();
	
	//use index array to hold and sort the indices
	int* gpuIndices;
	//initialize index array on GPU
	int numBlocks = (dim*dim + THREADS - 1) / THREADS;
	dim3 threadBlock(THREADS);
	dim3 blockGrid(numBlocks);
	cudaMalloc((void**)&gpuIndices, dim*dim*sizeof(int));

	//init indices array such that indices = [-1,1,2,...], whereas the diagonal elements
	//and the lower diagonal matrix contains -1. These entries should not be chosen
	//later because the matrix is symmetric.
	initIndicesKernel<<<blockGrid, threadBlock>>>(gpuIndices, dim);
	CUDA_CHECK_ERROR();

	int* indices = new int[dim*dim];
	cudaMemcpy(indices, gpuIndices, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);

	//sort array on GPU and sort indices array respectively
	thrust::sort_by_key(F, F + dim*dim, indices, thrust::greater<float>()); 
	CUDA_CHECK_ERROR();

    //free gpu memory
    cudaFree(gpuIndices);

	//get the kBest indices
	Indices* kBestIndices = new Indices[kBest];
	int countBest = 0;
	for(int i = 0; i < dim*dim && countBest < kBest; i++)
	{
		int idx = indices[i];
		int x = idx/dim;
		int y = idx%dim;

		if (-1 != idx && 0 == T->getVal(x,y)) //only upper diagonal matrix, where T is zero
		{
			kBestIndices[countBest].i = x;
			kBestIndices[countBest].j = y;
			countBest++;
		}
		//unused kBest-slots contain "-1" as indices per struct definition
	}

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

