/*
 * Initializer.cu
 *
 * Initializes a given directory by indexing all images within this directory
 * in a map such that the image-path can be obtained fast by the corresponding
 * index.
 * Initializes the T Matrix.
 *
 *  Created on: Jun 29, 2013
 *      Author: schwarzk
 */

#include "Initializer.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "Tester.h"
#include "GPUSparse.h"

#define CUDA_CHECK_ERROR() {							\
    cudaError_t err = cudaGetLastError();					\
    if (cudaSuccess != err) {						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
}

const int THREADS = 128;

//Initialize index arrays
static __global__ void initIndexArrays(int* d_idx1, int* d_idx2, int* d_res, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		d_idx1[idx] = idx;
		d_idx2[idx] = idx;
		d_res[idx] = 0;
	}
}

//Simple kernel to shuffle an array on GPU (prevents memcpy). To be called only with ONE thread!
static __global__ void shuffleArrayKernel(int* array, int dim, curandState* globalStates, unsigned long seed)
{
	int idx = threadIdx.x;
	curand_init(seed, idx, 0, &globalStates[idx]);
	curandState localState = globalStates[idx];
	for(int i = 0; i < dim; i++)
	{
		int shift = curand_uniform(&localState) * dim;
		int shift_idx = (i+shift)%dim;
		//shuffle
		int tmp = array[i];
		array[i] = array[shift_idx];
		array[shift_idx] = tmp;
	}
}

//Kernel to swap upper diagonal matrix elements
static __global__ void swapUpperDiagonal(int* idx1, int* idx2, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < dim)
	{
		//swap upper diagonal elements
		if (idx1[idx] <= idx2[idx])
		{
			int tmp = idx1[idx];
			idx1[idx] = idx2[idx];
			idx2[idx] = tmp;
		}
	}
}

//check for symmetric entries (duplicates caused by swapping earlier)
static __global__ void checkForDuplicates(int* idx1, int* idx2, int size, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (size-1))
	{
		if (idx1[idx] == idx2[idx]) //diagonal element (can not occur twice)
		{
			//mask out
			idx1[idx] = dim+1;
			idx2[idx] = dim+1;
		}
		else
		{
			int compareIdx = idx2[idx];
			for(int j = idx+1; j < size && idx1[idx] == idx1[j]; j++)
			{
				if (compareIdx == idx2[j])
				{
					//mask out
					idx1[idx] = dim+1;
					idx2[idx] = dim+1;
					break; //only one duplicate possible
				}
			}
		}
	}
}


Initializer::Initializer()
{

}

Initializer::~Initializer()
{

}

/*
 * Initializes the T-Matrix with random image comparisons.
 */
void Initializer::doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initArraySize)
{
	//Initialization arrays
	int* d_initIdx1;
	int* d_initIdx2;
	int* d_initRes;
	cudaMalloc(&d_initIdx1, initArraySize*sizeof(int));
	cudaMalloc(&d_initIdx2, initArraySize*sizeof(int));
	cudaMalloc(&d_initRes, initArraySize*sizeof(int));

	//initialize arrays
	int numBlocks = (initArraySize + THREADS - 1) / THREADS;
	dim3 threadBlock(THREADS);
	dim3 blockGrid(numBlocks);
	initIndexArrays<<<blockGrid, threadBlock>>>(d_initIdx1, d_initIdx2, d_initRes, initArraySize);
	CUDA_CHECK_ERROR()

	//todo remove debug printing
	int* testResult1 = new int[initArraySize];
	int* testResult2 = new int[initArraySize];
	int* testResult3 = new int[initArraySize];
	/*
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult3, d_initRes, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	Tester::printArrayInt(testResult3, initArraySize);
	*/

	//initialize cuRAND
	curandState* states;
	cudaMalloc(&states, sizeof(curandState));

	//shuffle Arrays
	//todo consider streaming?
	shuffleArrayKernel<<<1, 1>>>(d_initIdx1, initArraySize, states, time(NULL));
	CUDA_CHECK_ERROR()
	shuffleArrayKernel<<<1, 1>>>(d_initIdx2, initArraySize, states, 3*time(NULL)); //ensure different seeds
	CUDA_CHECK_ERROR()

	//todo remove debug printing
	/*
	printf("shuffled\n");
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	*/

	//mask each entry on the upper diagonal matrix (prevents symmetrical image comparisons)
	//swap entries inside upper diagonal-matrix (map to lower diagonal matrix)
	swapUpperDiagonal<<<blockGrid, threadBlock>>>(d_initIdx1, d_initIdx2, initArraySize);
	CUDA_CHECK_ERROR()

	//todo remove debug printing
	/*
	printf("swapped\n");
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	*/

	//sort index array1 ascending and index array2 respectively
	//wrap device pointers
	thrust::device_ptr<int> dp_initIdx1 = thrust::device_pointer_cast(d_initIdx1);
	thrust::device_ptr<int> dp_initIdx2 = thrust::device_pointer_cast(d_initIdx2);
	CUDA_CHECK_ERROR();
	thrust::sort_by_key(dp_initIdx1, dp_initIdx1 + initArraySize, dp_initIdx2); //ascending
	CUDA_CHECK_ERROR();

	//todo remove debug printing
	/*
	printf("sorted\n");
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	*/

	//check for duplicated enries and maks them out
	checkForDuplicates<<<blockGrid, threadBlock>>>(d_initIdx1, d_initIdx2, initArraySize, T->getDimension());
	CUDA_CHECK_ERROR();

	//todo remove debug printing
	/*
	printf("masked duplicates\n");
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	*/

	//sort again
	thrust::sort_by_key(dp_initIdx1, dp_initIdx1 + initArraySize, dp_initIdx2); //ascending
	CUDA_CHECK_ERROR();

	//todo remove debug printing
	/*
	printf("sorted\n");
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	*/

	//compare images
	cudaMemcpy(testResult3, d_initRes, initArraySize*sizeof(int), cudaMemcpyDeviceToHost); //todo remove me (just for testing)
	cudaMemcpy(testResult2, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost); //todo remove me (just for testing)
	comparator->doComparison(iHandler, T, testResult2, d_initIdx2, testResult3, initArraySize);
	cudaMemcpy(d_initRes, testResult3, initArraySize*sizeof(int), cudaMemcpyHostToDevice); //todo remove me (just for testing)

	//todo remove debug printing
	printf("result after comparison -> this will be given to GPUSparse to update the Marix\n");
	cudaMemcpy(testResult1, d_initIdx1, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_initIdx2, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult3, d_initRes, initArraySize*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, initArraySize);
	Tester::printArrayInt(testResult2, initArraySize);
	Tester::printArrayInt(testResult3, initArraySize);

	//initialize T Matrix with compared images
	//invoked only on sparse matrixhandler
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);
	T_sparse->updateSparseStatus(d_initIdx1, d_initIdx2, d_initRes, initArraySize);



	//cleanup after completed initialization
	cudaFree(d_initIdx1);
	cudaFree(d_initIdx2);
	cudaFree(d_initRes);
}


