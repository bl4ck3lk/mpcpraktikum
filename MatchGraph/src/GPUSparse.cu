/*
 * GPUSparse.cpp
 *
 *  Created on: Jun 12, 2013
 *      Author: Armin Gufler
 */

#include "GPUSparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Tester.h"
#include <thrust/sort.h>
#include <thrust/scan.h>

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

__global__ void rowPtrUpdateKernel(int* _rowPtr, int* _rowPtrIncr, int size)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < size)
	{
		_rowPtr[i] += _rowPtrIncr[i];
	}
}

//Scatters new elements in colIdx array; uses dim threads -> probably many threads have no work
__global__ void scatterNewElementsColIdx(int* dst, int* prefix, int* rowPtr, int* data, int dim)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	//i represents the row
	if (i < dim)
	{
		for (int s = prefix[i]; s < prefix[i + 1]; ++s)
			{
				dst[rowPtr[(i + 1)] + s] = data[s];
			}
	}
}

__global__ void colIdxIncrementKernel(int* colIdx, int* oldColIdx, int* rowPtr, int* incr, int size)
{
	//TODO can this kernel make profitable use of shared memory?

	unsigned int row = blockIdx.x;

	unsigned int rowIdx = threadIdx.x;

	for (int j = rowIdx; j < (rowPtr[row + 1] - rowPtr[row]); j += blockDim.x)
	{
 		colIdx[rowPtr[row] + j + incr[row]] = oldColIdx[rowPtr[row]+j];
	}
	//Note: a single thread has to do more than one step only if blockDim.x < length of row
	// Number of blocks has to be equal to number of rows for this kernel!
}


/***************************************************************************************/
/**************** KERNELS TO GET THE VALUE ARRAY ***************************************/
__global__ void scatterKernel(float* dst, int num)
{
	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	for(int pos = i; pos < num; pos += blockDim.x)
	{
		dst[pos] = (-1)*lambda_times_dim;
	}
	//TODO split into a known number of blocks, let each thread write one number

}

__global__ void scatterDiagonalKernel(float* gpuValues, int* gpuRowPtr,
		int* gpuDegrees, int* gpuDiagPos, int dim)
{
	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < dim)
	{
		int row = i;
		const int valueIndex = gpuRowPtr[row] + gpuDiagPos[row];
		const float valToWrite = 1 + (lambda_times_dim * gpuDegrees[row]);
		gpuValues[valueIndex] = valToWrite;
	}
}
/***************************************************************************************/
/***************************************************************************************/

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
 Adds the sum of each block to the specific block (for prefix sum)
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
	firstInitMode = true;

	rowPtr = (int*) malloc((dim+1)*sizeof(int));

	numNewDiagonal = 0;
	numNewSimilar = 0;
	num_dissimilar = 0;
	num_similar = 0;
	nnz_rows = 0;
	degrees = (int*) malloc((dim) * sizeof(int));
	std::fill_n(degrees, dim, 0);
	diagPos = (int*) malloc((dim) * sizeof(int));
	std::fill_n(diagPos, dim, 0);
}

GPUSparse::~GPUSparse()
{
	free(rowPtr);
	free(diagPos);
	free(degrees);
	free(colIdx);

	cudaFree(_gpuColIdx);
	cudaFree(_gpuRowPtr);
}

void GPUSparse::updateSparseStatus()
{


	bool testOutput = false;
	int old_nnz = getNNZ() - numNewDiagonal;
	if(testOutput)
		printf("nnz(old) = %i, new similar: %i , new diagonal %i \n", old_nnz, numNewSimilar, numNewDiagonal);

	//we have n new entries with value 1 -> n/2 similarities
	num_similar += (numNewSimilar / 2); //update number

	bool firstInit = firstInitMode;
	firstInitMode = false;

	int numNew = numNewSimilar + numNewDiagonal;

	int* newElemArr = (int*) malloc(numNew * sizeof(int));
	int* newElementRowArr = (int*) malloc(numNew * sizeof(int));
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
			const int row = (it->first);
			newElemArr[c] = (*lIter);
			newElementRowArr[c] = row;
			c++;
			rowPtrIncr[row + 1]++;
		}
		list.clear();
	}
	newElemMap.clear(); //reset the map for new elements

//	printf("new Elements are:  ");
//	Tester::printArrayInt(newElemArr, numNew);
// 	printf("rowPointerINCR: ");
// 	Tester::printArrayInt(rowPtrIncr, dim + 1);

	if (firstInit)
	{
		//column index array is just all new elements...
		colIdx = newElemArr;

		cudaMalloc((void**) &_gpuColIdx, numNew*sizeof(int));
		cudaMemcpy(_gpuColIdx, colIdx, numNew*sizeof(int), cudaMemcpyHostToDevice);

		
		cudaMalloc((void**) &_gpuRowPtr, (dim+1)*sizeof(int));
		cudaMemcpy(_gpuRowPtr, rowPtrIncr, (dim+1)*sizeof(int), cudaMemcpyHostToDevice);
		thrust::device_ptr<int> dev_ptr_row = thrust::device_pointer_cast(_gpuRowPtr);
		thrust::inclusive_scan(dev_ptr_row, dev_ptr_row+(dim+1), dev_ptr_row);
	  
		cudaMemcpy(rowPtr, _gpuRowPtr, (dim+1)*sizeof(int), cudaMemcpyDeviceToHost);
		
		//GPUSparse::prefixSumGPU(rowPtr, rowPtrIncr, dim+1);
	}
	else
	{
		printf("## Normal update starts \n");
// 		printGpuArray(_gpuRowPtr, dim+1, "111111111111 rowPtr (normal update)");
		
		//Tester::printArrayInt(colIdx, old_nnz);

		// variables to control thread and block size for kernel calls
		int numThreads = 1;
		int numBlocks = 1;
		
		/******* set up gpu device pointers **********/
		//prefix sum of row pointer increment
		int* _gpuRowPtrPrefix;
		cudaMalloc((void**) &_gpuRowPtrPrefix, (dim + 1) * sizeof(int));
		//new colDdx array
		int* _gpuColIdxNew;
		//array holding new elements (of column index)
		int* _gpuNewElements;
		cudaMalloc((void**) &_gpuNewElements, numNew*sizeof(int));

		//FIXME prefix sum computation seems to have problems with specific
		//configurations of thread and block size! Very sensitive...

		/* compute prefix sum of row pointer increment array
		 *(holds for every row the number of new elements) *************************************/
		const int prefix_dim = dim+1;
		
 		cudaMemcpy(_gpuRowPtrPrefix, rowPtrIncr, prefix_dim * sizeof(int), cudaMemcpyHostToDevice);
		
		thrust::device_ptr<int> dev_ptr_prefix = thrust::device_pointer_cast(_gpuRowPtrPrefix);
		thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix+prefix_dim, dev_ptr_prefix);
		
		printf("prefix sum (thrust) done...\n");
		
// 		int* _gpuSumOfBlocks;
// 		cudaMalloc((void**) &_gpuSumOfBlocks, numBlocks * sizeof(int));
// 
// 		printGpuArray(_gpuRowPtr, dim+1, "2222222222222 rowPtr (normal update)");
// 		
// 		prefixSumKernel<<<numBlocks, numThreads>>>(_gpuRowPtrPrefix, _gpuSumOfBlocks, prefix_dim, 1);
// 		
// 		printGpuArray(_gpuRowPtr, dim+1, "3333333333333 rowPtr (normal update)");
// 		
// 
// 		prefixSumKernel<<<1, numThreads>>>(_gpuSumOfBlocks, _gpuSumOfBlocks, numBlocks, 0);
// 		
// 		printGpuArray(_gpuRowPtr, dim+1, "44444444444444 rowPtr (normal update)");
// 		
// 
// 		addKernel<<<numBlocks, numThreads>>>(_gpuRowPtrPrefix, _gpuSumOfBlocks);
// 		
// 		printGpuArray(_gpuRowPtr, dim+1, "55555555555555 rowPtr (normal update)");
// 		
// 
// 		cudaFree(_gpuSumOfBlocks);
// 
// 		printf("==========PREFIX SUM OF ROW POINTER");
// 		int* test = (int*) malloc(sizeof(int)*prefix_dim);
// 		cudaMemcpy(test, _gpuRowPtrPrefix, sizeof(int)*prefix_dim, cudaMemcpyDeviceToHost);
// 		Tester::printArrayInt(test, prefix_dim);
		/***************************************************************************************/

		
// 		printGpuArray(_gpuRowPtr, dim+1, "66666666666666 rowPtr (normal update)");
		
		//TODO can we hold the colIdx array on GPU all the time?

		//the current colIdx array
		int* _gpuColIdxOld;
		cudaMalloc((void**) &_gpuColIdxOld, old_nnz * sizeof(int));
		cudaMemcpy(_gpuColIdxOld, colIdx, old_nnz * sizeof(int), cudaMemcpyHostToDevice);

		//FIXME should be kept on GPU
//				cudaMalloc((void**) &_gpuRowPtr, (dim+1)*sizeof(int));
//				cudaMemcpy(_gpuRowPtr, rowPtr, dim+1, cudaMemcpyHostToDevice);
//				printGpuArray(_gpuRowPtr, dim+1, "rowPtr (GPU, after UPLOAD)");
//				printf("!!!!!!!!!!!!row pointer CPU !!!!!!!!!!!!!!!!!!!\n");
//				Tester::printArrayInt(rowPtr, dim+1);

		cudaMalloc((void**) &_gpuColIdxNew, (old_nnz + numNew) * sizeof(int));

//		printGpuArray(_gpuColIdxOld, old_nnz, "Old colIdx");

		//old column index values are shifted to new locations, according to number of new elements inserted
		numThreads = 256;
		numBlocks = dim;
		colIdxIncrementKernel<<<numBlocks,numThreads>>>(_gpuColIdxNew, _gpuColIdxOld, _gpuRowPtr, _gpuRowPtrPrefix, old_nnz);
		
		printf("colIdxIncrementKernel done...\n");
		
		//copy new elements to gpu
		cudaMemcpy(_gpuNewElements, newElemArr, numNew * sizeof(int), cudaMemcpyHostToDevice);
		//new elements are written into column index array (unsorted, at end of row)
		numThreads = 256;
		numBlocks = 1 + (dim/numThreads);

// 		printGpuArray(_gpuColIdxNew, (old_nnz+numNew), "Intermediate new colIdx");
// 		printGpuArray(_gpuNewElements, (numNew), "New GPU elements");
// 		printf("New elements ROW : ");
// 		Tester::printArrayInt(newElementRowArr, numNew);

		scatterNewElementsColIdx<<<numBlocks,numThreads>>>(_gpuColIdxNew, _gpuRowPtrPrefix, _gpuRowPtr, _gpuNewElements, dim);
	
		cudaFree(_gpuNewElements);
		cudaFree(_gpuColIdxOld);

// 		printGpuArray(_gpuColIdxNew, (old_nnz+numNew), "Final new colIdx");
// 
// 		printGpuArray(_gpuRowPtrPrefix, dim+1, "rowPtrPrefix (GPU)");
// 		printGpuArray(_gpuRowPtr, dim+1, "rowPtr (GPU, before update kernel)");

		// compute the updated row pointer now
		numThreads = 256;
		numBlocks = 1 + (dim/numThreads);

		rowPtrUpdateKernel<<<numBlocks, numThreads>>>(_gpuRowPtr, _gpuRowPtrPrefix , dim+1);

		printf("all Kernels done...\n");
		
		cudaMemcpy(rowPtr, _gpuRowPtr, (dim+1) * sizeof(int), cudaMemcpyDeviceToHost);
		
		cudaFree(_gpuRowPtrPrefix);

		cudaDeviceSynchronize();

// 		printGpuArray(_gpuRowPtr, dim+1, "rowPr (GPU)");
// 		Tester::printArrayInt(rowPtr, dim+1);

		//set up device pointer for thrust library
		thrust::device_ptr<int> dev_ptr_colIdx = thrust::device_pointer_cast(_gpuColIdxNew);
		int start;
		int end;
		//sort colIdx row-wise using thrust library
		for(int r = 0; r < dim; ++r)
		{
		  if(rowPtrIncr[r+1] == 0)
			continue;

		  start = rowPtr[r];
		  end = rowPtr[r+1];
		  thrust::sort(dev_ptr_colIdx + start, dev_ptr_colIdx + end);
		}
		
		printf("Sorting (thrust) done...\n");

		int* colIdxNew = (int*) malloc((old_nnz + numNew) * sizeof(int));
		cudaMemcpy(colIdxNew, _gpuColIdxNew, (old_nnz + numNew) * sizeof(int), cudaMemcpyDeviceToHost);

		free(colIdx);
		free(newElemArr);

		cudaFree(_gpuColIdx);
		_gpuColIdx = _gpuColIdxNew;
		colIdx = colIdxNew;
	}

	free(newElementRowArr);
	free(rowPtrIncr);

	if(testOutput)
	{
		/********* TESTING ****************/
		printf("AFTER UPDATE ---> num similar = %i nnz = %i \n", num_similar, getNNZ());
		printf("colIdx after update: ");
		Tester::printArrayInt(colIdx, getNNZ());
		printf("rowPtr after update: ");
		Tester::printArrayInt(rowPtr, dim+1);
		float* vals = getValueArr(false);
		free(vals);
		/**********************************/
	}


	numNewSimilar = 0; //reset
	numNewDiagonal = 0; //reset
}

void GPUSparse::set(int i, int j, bool val)
{ //NOTE handle similarities

	if(i >= dim || j >= dim)
		return;

	if (val)
	{ 
		  // a new 1 to set within matrix
		  //increment degree
			  //printf("set (%i, %i) \n", i, j);
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
	//first, check if this index (i,j) is already set to 1
 	IndexMap::iterator iter = similarMap.find(row);
		if(iter != similarMap.end())
		{
		  if((iter->second.find(j)) == (iter->second.end()))
		  {
			iter->second.insert(j);
		  }
		  else
		  {
			return;
		  }
		}
		else
		{
			boost::unordered_set<int> jSet;
			jSet.insert(j);
			similarMap.insert(IndexMap::value_type(row, jSet));
		}
	

	bool insertedSuccessfully = false;
	
	//second, look if this element is already staged for insertion
	myElemMap::iterator it = newElemMap.find(row);
			
	if (it == newElemMap.end())
	{ //row not in map
	  std::set<int> list;
	  list.insert(j);
	  newElemMap.insert(std::pair<int, std::set<int> >(row, list));
	  insertedSuccessfully = true;
	}
	else
	{
	  if((it->second.insert(j)).second)
		insertedSuccessfully = true;
	}
			

	if(insertedSuccessfully)
	{
	  //insertion was really executed, therefore update some status variables
	  if (row != j)
		{ //it is not a diagonal element
			numNewSimilar++;
			if(j < row)
				diagPos[row]++; //diagonal element now 1 position to the right
		}
		else
		{ //it is a diagonal element
			numNewDiagonal++;
			nnz_rows++;
		}
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

float* GPUSparse::getValueArr(bool gpuPointer) const
{
	// Attention: blockGrid() and threadBlock() does corrupt memory
	//dim3 blockGrid(NUM_BLOCKS);
	//dim3 threadBlock(MAX_THREADS);

	int nnz = getNNZ();

	float _cpuLambda_times_dim = dim * lambda; //FIXME do not compute each time
	cudaMemcpyToSymbol(lambda_times_dim, &_cpuLambda_times_dim, sizeof(float));

	float* gpuValues;
	cudaMalloc((void**) &gpuValues, nnz * sizeof(float));
	int NUM_THREADS = 512;
	int NUM_BLOCKS = 1;
	scatterKernel<<<NUM_BLOCKS, NUM_THREADS>>>(gpuValues, nnz);

	int* gpuDegrees;
	cudaMalloc((void**) &gpuDegrees, dim * sizeof(int));
	cudaMemcpy(gpuDegrees, degrees, dim * sizeof(int), cudaMemcpyHostToDevice);

	int* gpuDiagPos;
	cudaMalloc((void**) &gpuDiagPos, dim * sizeof(int));
	cudaMemcpy(gpuDiagPos, diagPos, dim * sizeof(int), cudaMemcpyHostToDevice);

	NUM_THREADS = 512;
	NUM_BLOCKS = 1 + (dim / NUM_THREADS);
	scatterDiagonalKernel<<<NUM_BLOCKS, NUM_THREADS>>>(gpuValues, _gpuRowPtr, gpuDegrees, gpuDiagPos, dim);

	cudaFree(gpuDiagPos);
	cudaFree(gpuDegrees);

	if (!gpuPointer)
	{
		float* valuesCPU = (float*) malloc(nnz * sizeof(float));
		cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(gpuValues);

		//printf("value array: ");
		//Tester::printArrayFloat(valuesCPU, nnz);

		return valuesCPU;
	}

	return gpuValues;
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
	printf("###INFO about SPARSE MATRIX###\n");
	printf("dim = %i, lambda = %f, nnz = %i \n", dim, lambda, getNNZ());
	printf("coldIdx (on CPU) : \n");
	Tester::printArrayInt(colIdx, getNNZ());
	printGpuArray(_gpuRowPtr, dim+1, "rowPtr (on GPU)");
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

int* GPUSparse::getRowPtrDevice() const
{
	return _gpuRowPtr;
}

int* GPUSparse::getColIdx() const
{
	return colIdx;
}

int* GPUSparse::getColIdxDevice() const
{
	return _gpuColIdx;
}

unsigned int GPUSparse::getNNZ() const
{
	return num_similar * 2 + nnz_rows;
}

void GPUSparse::setRandom(int num)
{
	//printf("begin to set %i random values...\n", num);
  
	int counter = num;
	while (counter-- > 0)
	{
		int randI = rand() % dim;
		int randJ = rand() % dim;

		if(randI == randJ)
		{
			//printf("EQUAL %i\n", randI);
			counter++;
			continue;
		}

		if (((double)rand() / (double)RAND_MAX ) < .4)
		{
			set(randI, randJ, true);
		}
		else
		{
			set(randI, randJ, false);
		}
	}
	
	//printf("Finished setting random values!\n");

}

int* GPUSparse::prefixSumGPU(int* result, const int* array, const int dimension)
{
	const int MAX_THREADS = 512;
	const int NUM_BLOCKS = 256;

	int* _gpuArrayData;
	cudaMalloc((void**) &_gpuArrayData, (dimension) * sizeof(int));
	cudaMemcpy(_gpuArrayData, array, (dimension) * sizeof(int),
			cudaMemcpyHostToDevice);

	int* _gpuSumOfBlocks;
	cudaMalloc((void**) &_gpuSumOfBlocks, NUM_BLOCKS * sizeof(int));

	prefixSumKernel<<<NUM_BLOCKS, MAX_THREADS>>>(_gpuArrayData, _gpuSumOfBlocks, dimension, 1);

	prefixSumKernel<<<1, MAX_THREADS>>>(_gpuSumOfBlocks, _gpuSumOfBlocks, NUM_BLOCKS, 0);

	addKernel<<<NUM_BLOCKS, MAX_THREADS>>>(_gpuArrayData, _gpuSumOfBlocks);

	cudaMemcpy(result, _gpuArrayData, dimension * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(_gpuSumOfBlocks);
	cudaFree(_gpuArrayData);
	
	return NULL;
}

void GPUSparse::printGpuArray(int * devPtr, const int size, std::string message)
{
	int* cpu = (int*) malloc(sizeof(int)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << message << " : ";
	Tester::printArrayInt(cpu, size);

}
