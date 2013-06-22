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
#include "Tester.h"

#define CUDA_SAFE_CALL(err) {							\
if (cudaSuccess != err)	{						\
    fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
            __FILE__, __LINE__, cudaGetErrorString(err) );	\
    exit(EXIT_FAILURE);						\
}									\
}

#define CUDA_CHECK_ERROR() {							\
cudaError_t err = cudaGetLastError();					\
if (cudaSuccess != err) {						\
    fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
            __FILE__, __LINE__, cudaGetErrorString(err) );	\
    exit(EXIT_FAILURE);						\
}									\
}

__device__ __constant__ float lambda_times_dim;

__global__ void scatterKernel(float* dst, int num)
{
	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < num)
		dst[i] = -lambda_times_dim;
}

__global__ void scatterDiagonalKernel(float* gpuValues, int* gpuRowPtr,
		int* gpuDegrees, int* gpuDiagPos, int dim)
{
	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	for (int row = i; row < dim; row += blockDim.x)
	{
		const int valueIndex = gpuRowPtr[row] + gpuDiagPos[row];
		const float valToWrite = 1 + (lambda_times_dim * gpuDegrees[row]);
		gpuValues[valueIndex] = valToWrite;
	}

}

//mode = 0 => dont save sum of each block (to use the same kernel for addition of sumOfBlocks)
//mode = 1 => save sum of each block
__global__ void prefixSumKernel(int* scanArray, int* sumOfBlocks, int _w,
		int mode)
{
	int idx;

	/* BLOCKWISE */
	//compaction
	int stride = 1;
	while (stride < blockDim.x)
	{
		idx = blockIdx.x * blockDim.x + (threadIdx.x + 1) * stride * 2 - 1;
		if (idx < (blockDim.x + blockIdx.x * blockDim.x))
		{
			scanArray[idx] += scanArray[idx - stride];

			//Store sum of blocks in auxiliary array
			if ((idx + 1) % blockDim.x == 0 && mode)
			{
				sumOfBlocks[blockIdx.x] = scanArray[idx];
			}
		}

		stride *= 2;
		__syncthreads();
	}
	__syncthreads();

	//spreading
	stride = blockDim.x >> 1;
	while (stride > 0)
	{
		idx = blockIdx.x * blockDim.x + (threadIdx.x + 1) * stride * 2 - 1;
		if ((idx + stride) < (blockDim.x + blockIdx.x * blockDim.x))
		{
			int tmp = idx + stride;
			if (tmp < _w)
			{
				scanArray[tmp] += scanArray[idx];
			}
		}
		stride >>= 1;
		__syncthreads();
	}
	__syncthreads();
	/* scanArray contains BLOCKWISE prefixsum */
}

/*
 Adds the sum of each block to the specific block
 */
__global__ void addKernel(int* scanArray, int* sumOfBlocks)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x > 0)
	{
		scanArray[idx] += sumOfBlocks[blockIdx.x - 1];
	}
}

GPUSparse::GPUSparse(unsigned int _dim, float _lambda) :
		dim(_dim), lambda(_lambda)
{
	rowPtr = (int*) malloc((dim + 1) * sizeof(int));
	colIdx = NULL;
	numNewDiagonal = 0;
	numNewSimilar = 0;
	num_dissimilar = 0;
	num_similar = 0;
	degrees = (int*) malloc((dim) * sizeof(int));
	std::fill_n(degrees, dim, 0);
	diagPos = (int*) malloc((dim) * sizeof(int));
	std::fill_n(diagPos, dim, 0);

	if (false)
	{
		int testColIdx[] =
		{ 0, 1, 0, 1, 2, 3, 4, 2, 3, 2, 4 };
		int testPtr[] =
		{ 0, 2, 4, 7, 9, 11 };

		degrees[0] = 1;
		degrees[1] = 1;
		degrees[2] = 2;
		degrees[3] = 1;
		degrees[4] = 1;

		num_similar = 3;
		colIdx = (int*) malloc((dim + num_similar * 2) * sizeof(int));
		for (int i = 0; i < (num_similar * 2 + dim); ++i)
		{
			colIdx[i] = testColIdx[i];
		}
		for (int i = 0; i < dim + 1; ++i)
		{
			rowPtr[i] = testPtr[i];
		}
		printf("Test initialization:\n");
		Tester::printArrayInt(rowPtr, dim + 1);
		Tester::printArrayInt(colIdx, num_similar * 2 + dim);
	}
}

void GPUSparse::initFirst()
{
	//we have n new entries with value 1 -> n/2 similarities
	num_similar = numNewSimilar / 2;

	//settin put column index array
	colIdx = (int*) malloc((dim + num_similar * 2) * sizeof(int));
}

void GPUSparse::updateSparseStatus()
{
	//printf("val array before update:");
	//getValueArr();

	printf("new similar: %i , new diagonal %i \n", numNewSimilar, numNewDiagonal);

	bool firstInit = (colIdx == NULL);
	if (numNewSimilar != 0 && firstInit)
	{ //first initialization
		initFirst();
	}

	int numNew = numNewSimilar + numNewDiagonal;

	int nnz = num_similar * 2 + dim;

	std::cout << "new Elements: " << numNew << std::endl;


	int* newElemArr = (int*) malloc(numNew * sizeof(int));
	int* rowPtrIncr = (int*) malloc((dim + 1) * sizeof(int));
	std::fill_n(rowPtrIncr, dim + 1, 0);

	int c = 0;
	for (myElemMap::iterator it = newElemMap.begin(); it != newElemMap.end();
			++it)
	{
		std::set<int> list = it->second;
		for (std::set<int>::const_iterator lIter = list.begin();
				lIter != list.end(); ++lIter)
		{
			newElemArr[c] = (*lIter);
			c++;
			rowPtrIncr[(it->first) + 1]++;
		}
	}

	printf("new Elements are:  ");
	Tester::printArrayInt(newElemArr, numNew);
	printf("rowPointerIncrement: ");
	Tester::printArrayInt(rowPtrIncr, dim + 1);

	if (firstInit)
	{
		//column index are just all new elements...
		colIdx = newElemArr;

		GPUSparse::prefixSumGPU(rowPtr, rowPtrIncr, dim+1);

		printf("initialized rowPtr: ");
		Tester::printArrayInt(rowPtr, dim + 1);
	}

	Tester::printArrayInt(degrees, dim);
	Tester::printArrayInt(diagPos, dim);
	printf("num similar %d \n", num_similar);
	getValueArr(false, NULL, NULL, NULL);

	printf("column B \n");
	Tester::printArrayFloat(getColumn(3),dim);

	if(!firstInit){
		int* rowPtrPrefixSum = (int*) malloc((dim+1)*sizeof(int));
		GPUSparse::prefixSumGPU(rowPtrPrefixSum, rowPtr, dim+1);
		printf("prefixSum: ");
		Tester::printArrayInt(rowPtrPrefixSum, dim + 1);

		//build new larger column index array
		int* colIdxNew = (int*) malloc((nnz + numNew) * sizeof(int));
		std::fill_n(colIdxNew, nnz + numNew, -1);

		//scatter new elements
		for (int i = 0; i < dim; ++i)
		{
			if (rowPtrIncr[i + 1] > 0)
			{
				//printf(" iterating from %i : %i (for %i) \n", rowPtrPrefixSum[i], rowPtrPrefixSum[i+1], i);
				int thisBucket = 0;
				for (int s = rowPtrPrefixSum[i]; s < rowPtrPrefixSum[i + 1];
						++s, ++thisBucket)
				{
					colIdxNew[rowPtr[(i + 1)] + s] = newElemArr[s];
					//printf(" %i goes to %i \n", ne[s], rowPtr[(i+1)]+s);
				}
			}
		}

		//scatter old elements
		for (int i = 0; i < dim; ++i)
		{
			//printf(" iterating from %i : %i (for %i) \n", rowPtr[i], rowPtr[i+1], i);
			for (int s = rowPtr[i]; s < rowPtr[i + 1]; ++s)
			{
				colIdxNew[s + rowPtrPrefixSum[i]] = colIdx[s];
			}
		}
		Tester::printArrayInt(colIdxNew, nnz + numNew);

		/* compute the updated row pointer now */
		int newRowPtr[] =
		{ 0, 3, 6, 10, 12, 15 };
		free(rowPtr);
		free(colIdx);

		rowPtr = newRowPtr;
		colIdx = colIdxNew;
	}

	free(rowPtrIncr);
	free(newElemArr);

	num_similar += numNewSimilar; //update number
	numNewSimilar = 0; //reset
	numNewDiagonal = 0; //reset
}

void GPUSparse::set(int i, int j, bool val)
{ //NOTE handle similarities

	if (val)
	{ // a new 1 to set within matrix
	  //increment degree
		incrementDegree(i);
		incrementDegree(j);

		addNewToRow(i, j);
		addNewToRow(j, i);
	}
	else
	{ // a -1 to add
		addDissimilarToColumn(i, j);
		addDissimilarToColumn(j, i);
	}

	if (i == 1 && j == 4)
		updateSparseStatus();
}

void GPUSparse::addDissimilarToColumn(int column, int row)
{
	myElemMap::iterator it = dissimilarMap.find(column);

	if (it == dissimilarMap.end())
	{ //row not in map
		std::set<int> list;
		list.insert(row);
		dissimilarMap.insert(std::pair<int, std::set<int> >(column, list));
	}
	else
	{
		it->second.insert(row);
	}
}

void GPUSparse::addNewToRow(int row, int j)
{
	if (row != j)
	{ //it is not a diagonal element
		numNewSimilar++;
		if(j < row)
			diagPos[row]++; //diagonal element now 1 position to the right
	}
	else
	{
		numNewDiagonal++;
	}


	myElemMap::iterator it = newElemMap.find(row);

	if (it == newElemMap.end())
	{ //row not in map
		std::set<int> list;
		list.insert(j);
		newElemMap.insert(std::pair<int, std::set<int> >(row, list));
	}
	else
	{
		it->second.insert(j);
	}
}

void GPUSparse::incrementDegree(int row)
{
	if (degrees[row] == 0)
	{ //first neighbor of this node
	  //state that at this position now not sparse
		addNewToRow(row, row);
	}
	degrees[row]++;
}

unsigned int GPUSparse::getDimension()
{
	return dim;
}

float* GPUSparse::getConfMatrixF()
{
	return NULL;
}

float* GPUSparse::getValueArr(bool gpuPointer, float* _gpuVals, int* _gpuRowPtr, int* _gpuColIdx) const
{
	const int MAX_THREADS = 128;
	const int NUM_BLOCKS = 256;
	dim3 blockGrid(NUM_BLOCKS);
	dim3 threadBlock(MAX_THREADS);

	int nnz = num_similar * 2 + dim;

	float _cpuLambda_times_dim = dim * lambda; //FIXME do not compute each time
	cudaMemcpyToSymbol(lambda_times_dim, &_cpuLambda_times_dim, sizeof(float));

	float* gpuValues;
	cudaMalloc((void**) &gpuValues, nnz * sizeof(float));

	scatterKernel<<<blockGrid, threadBlock>>>(gpuValues, nnz);
	int* gpuDegrees;
	cudaMalloc((void**) &gpuDegrees, dim * sizeof(int));
	cudaMemcpy(gpuDegrees, degrees, dim * sizeof(int), cudaMemcpyHostToDevice);
	int* gpuRowPtr;
	cudaMalloc((void**) &gpuRowPtr, (dim + 1) * sizeof(int));
	cudaMemcpy(gpuRowPtr, rowPtr, (dim + 1) * sizeof(int),
			cudaMemcpyHostToDevice);
	int* gpuDiagPos;
	cudaMalloc((void**) &gpuDiagPos, dim * sizeof(int));
	cudaMemcpy(gpuDiagPos, diagPos, dim * sizeof(int), cudaMemcpyHostToDevice);
	scatterDiagonalKernel<<<blockGrid, threadBlock>>>(gpuValues, gpuRowPtr,
			gpuDegrees, gpuDiagPos, dim);


	if(!gpuPointer){
		float* valuesCPU = (float*) malloc(nnz * sizeof(float));
		cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(float),
				cudaMemcpyDeviceToHost);


		printf("value array: ");
		Tester::printArrayFloat(valuesCPU, nnz);

		return valuesCPU;
	}

	_gpuRowPtr = gpuRowPtr;
	_gpuVals = gpuValues;

	_gpuColIdx = NULL; //TODO?

	return NULL;
}

float* GPUSparse::getColumn(int columnIdx) const
{
	float* column = new float[dim];
	std::fill_n(column, dim, 0.0f);

	myElemMap::const_iterator it = dissimilarMap.find(columnIdx);

	if (it != dissimilarMap.end())
	{
		std::set<int> dis = it->second;
		for (std::set<int>::const_iterator lIter = dis.begin();
				lIter != dis.end(); ++lIter)
		{
			int idx = (*lIter);
			column[idx] = -1.0f;
		}
	}

	for(int i = rowPtr[columnIdx]; i < rowPtr[columnIdx+1]; i++)
	{
		column[colIdx[i]] = 1.0f;
	}

	//diagonal element
	column[columnIdx] = 0.0f;

	return column;
}

char* GPUSparse::getMatrAsArray()
{
	return NULL;
} //TODO

char GPUSparse::getVal(int i, int j)
{
	return 'c'; //TODO
}

int GPUSparse::getSimilarities()
{
	return 0;
}

void GPUSparse::print()
{
	//TODO
}

void GPUSparse::writeGML(char * filename, bool similar, bool dissimilar,
		bool potential)
{
	//TODO
}

int* GPUSparse::getRowPtr() const
{
	return rowPtr;
}

int* GPUSparse::getColIdx() const
{
	return colIdx;
}

unsigned int GPUSparse::getNNZ() const
{
	return num_similar * 2 + dim;
}

void GPUSparse::prefixSumGPU(int* result, const int* array, const int dimension)
{
	const int MAX_THREADS = 128;
	const int NUM_BLOCKS = 256;
	dim3 blockGrid(NUM_BLOCKS);
	dim3 threadBlock(MAX_THREADS);

	int* _gpuArrayData;
	cudaMalloc((void**) &_gpuArrayData, (dimension) * sizeof(int));
	cudaMemcpy(_gpuArrayData, array, (dimension) * sizeof(int),
			cudaMemcpyHostToDevice);

	int* _gpuSumOfBlocks;
	cudaMalloc((void**) &_gpuSumOfBlocks, NUM_BLOCKS * sizeof(int));

	prefixSumKernel<<<blockGrid, threadBlock>>>(_gpuArrayData, _gpuSumOfBlocks,
			dimension, 1);

	prefixSumKernel<<<1, threadBlock>>>(_gpuSumOfBlocks, _gpuSumOfBlocks,
			NUM_BLOCKS, 0);

	addKernel<<<blockGrid, threadBlock>>>(_gpuArrayData, _gpuSumOfBlocks);

	cudaMemcpy(result, _gpuArrayData, dimension * sizeof(int),
			cudaMemcpyDeviceToHost);

	cudaFree(_gpuArrayData);
	cudaFree(_gpuSumOfBlocks);
}
