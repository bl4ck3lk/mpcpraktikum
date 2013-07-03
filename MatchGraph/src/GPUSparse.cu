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

__device__ int deviceVar;

__global__ void arrayAddKernel(int* res, int* _a1, int* _a2, int size)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < size)
	{
		res[i] = _a1[i] + _a2[i];
	}
}

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

__global__ void rowIncrementedKernel(int* dst, int* idxArray1, int* idxArray2, int dstSize, int idxSize)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < idxSize)
	{
		int curIdx = idxArray1[i];
		atomicAdd(&dst[curIdx+1], 1);
		curIdx = idxArray2[i];
		atomicAdd(&dst[dstSize+curIdx+1], 1);
	}
}

__global__ void initKernel(int * array, const int val, const int nwords)
{
    int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    for(; tIdx < nwords; tIdx += blockDim.x)
        array[tIdx] = val;
}

__global__ void cleanIndexArrays(int* cleanIdx1, int* cleanIdx2, int* negIdx, int* idx1, int* idx2, int* res, int* prefix, int k)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

	if(tIdx < k)
	{
		if(res[tIdx] != 0)
		{
			const int idxToWrite = prefix[tIdx];
			cleanIdx1[idxToWrite] = idx1[tIdx];
			cleanIdx2[idxToWrite] = idx2[tIdx];
		}
		else
		{
			const int idxToWrite = (tIdx - prefix[tIdx])*2;
			negIdx[idxToWrite] = idx1[tIdx];
			negIdx[idxToWrite+1] = idx2[tIdx];
		}
	}
}

__global__ void doInsertionKernel(int* colIdx, int* rowPtr, int* oldRowPtr, int* idxData, int* idxPrefix, int dim)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;
	
	//using one thread per row
	//each thread simply applies insertion sort for each element (sequential)

	if(tIdx < dim)
	{
	  const int start = idxPrefix[tIdx];
	  const int end = idxPrefix[tIdx+1];
	  
	  int counter = 0;
	  for(int i = start; i < end; ++i, counter++)
	  {
		const int elem = idxData[i];
		int j = rowPtr[tIdx] + counter + (oldRowPtr[tIdx+1]-oldRowPtr[tIdx]);
		while(j > rowPtr[tIdx] && colIdx[j-1] > elem)
		{
			colIdx[j] = colIdx[j-1];
			j--;
		}
		  colIdx[j] = elem;
	  }
	}
}

__global__ void initColIdxRowPtrKernel(int* colIdx, int* rowPtr, int dim)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if(tIdx < dim)
	{
		colIdx[tIdx] = tIdx;
		rowPtr[tIdx] = tIdx;
	}
	if(tIdx == 0)
		rowPtr[dim] = dim;
}

__global__ void updateDegreeKernel(int* degrees, int* incr, int dim)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if (tIdx < dim)
	{
		degrees[tIdx] += incr[tIdx+1];
	}
}

__global__ void updateDiagPosKernel(int* diagPos, int* idx1, int* idx2, int size)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if(tIdx < size)
	{
		int col = idx2[tIdx];
		int row = idx1[tIdx];
		if(col < row)
			atomicAdd(&diagPos[row], 1);

		const int tmpSwap = col;
		col = row;
		row = tmpSwap;
		if(col < row)
			atomicAdd(&diagPos[row], 1);
	}
}

__global__ void columnWriteKernel(float* dst, int* colIdx, int* rowPtr, int* numOnes, int col, int dim)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

	if(tIdx == 0){
		//put into 'shared' memory the amount of 1's of this column
		numOnes[0] = rowPtr[col+1] - rowPtr[col];
	}
	__syncthreads();

	for(int i = tIdx; i < numOnes[0] && i < dim; i+=blockDim.x)
	{
		dst[colIdx[rowPtr[col+i]]] = 1.0;
	}

	__syncthreads();
	if(tIdx == 0)
	{
		//diagonal to zero
		dst[col] = 0.0;
	}

	//	if (tIdx < numOnes[0] && tIdx < dim)
//	{
//		dst[rowPtr[col+tIdx]] = 1.0;
//	}
}

GPUSparse::GPUSparse(unsigned int _dim, float _lambda) :
		dim(_dim), lambda(_lambda)
{
	firstInitMode = true;
	num_dissimilar = 0;
	num_similar = 0;
	nnz_rows = 0;

	cudaMalloc((void**) &_gpuRowPtr, (dim+1)*sizeof(int));
	cudaMalloc((void**) &_gpuColIdx, dim*sizeof(int));
	cudaMalloc((void**) &_gpuDegrees, dim*sizeof(int));
	cudaMalloc((void**) &_gpuDiagPos, dim*sizeof(int));

	const int numThreads = 128;
	const int numBlocks = 1 + (dim/numThreads);
	initColIdxRowPtrKernel<<<numBlocks, numThreads>>>(_gpuColIdx, _gpuRowPtr, dim);
	initKernel<<<numBlocks, numThreads>>>(_gpuDegrees, 0, dim);
	initKernel<<<numBlocks, numThreads>>>(_gpuDiagPos, 0, dim);
	nnz_rows = dim;
}

GPUSparse::~GPUSparse()
{

}

void GPUSparse::handleDissimilar(int* idxData, int num)
{
	for(int i = 0; i < num*2; i+=2)
	{
		int col = idxData[i];
		int row = idxData[i+1];
		addDissimilarToColumn(col, row);
		addDissimilarToColumn(row, col);
	}
}

void GPUSparse::updateSparseStatus(int* _idx1, int* _idx2, int* _res, int _k)
{
	int* idx1;
	int* idx2;
	int* res;
	int k = 7;

	printf("Current Matrix:\n");
	printGpuArray(_gpuColIdx, getNNZ(), "colIdx");
	printGpuArray(_gpuRowPtr, dim + 1, "rowPtr");
	printGpuArray(_gpuDiagPos, dim, "diagPos");
	printGpuArray(_gpuDegrees, dim, "degrees");

	if (true)
	{
		std::cout << "First Test Initialization" << std::endl;
		cudaMalloc((void**) &idx1, k * sizeof(int));
		cudaMalloc((void**) &idx2, k * sizeof(int));
		cudaMalloc((void**) &res, k * sizeof(int));
		int Tidx1[7] =
		{ 0, 0, 1, 1, 3, 3, 4 };
		int Tidx2[7] =
		{ 3, 1, 4, 2, 2, 1, 2 };
		int Tres[7] =
		{ 1, 1, 0, 1, 1, 0, 1 };
		cudaMemcpy(idx1, Tidx1, k * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(idx2, Tidx2, k * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(res, Tres, k * sizeof(int), cudaMemcpyHostToDevice);
	}

	printf("input:\n");
	printGpuArray(idx1, k, "idx1");
	printGpuArray(idx2, k, "idx2");
	printGpuArray(res, k, "res ");
	printf("\n");


	int numThreads;
	int numBlocks;

	int* prefixSumResult;
	cudaMalloc((void**) &prefixSumResult, (1 + k) * sizeof(int));
	thrust::device_ptr<int> dev_ptr_prefix_res = thrust::device_pointer_cast(prefixSumResult);
	thrust::device_ptr<int> dev_ptr_prefix = thrust::device_pointer_cast(res);
	thrust::exclusive_scan(dev_ptr_prefix, dev_ptr_prefix + k + 1, dev_ptr_prefix_res);

	printGpuArray(prefixSumResult, k + 1, "prefix result");

	int numSimilar;
	cudaMemcpy(&numSimilar, prefixSumResult + k, sizeof(int), cudaMemcpyDeviceToHost);
	int* cleanedIdx1;
	int* cleanedIdx2;
	int* negativeIndx;
	cudaMalloc((void**) &cleanedIdx1, numSimilar * sizeof(int));
	cudaMalloc((void**) &cleanedIdx2, numSimilar * sizeof(int));
	cudaMalloc((void**) &negativeIndx, (2*(k - numSimilar))*sizeof(int));

	numThreads = 32;
	numBlocks = 1 + (k / numThreads);
	cleanIndexArrays<<<numBlocks, numThreads>>>(cleanedIdx1, cleanedIdx2, negativeIndx, idx1, idx2, res, prefixSumResult, k);

	printGpuArray(cleanedIdx1, numSimilar, "cleaned Idx1");
	printGpuArray(cleanedIdx2, numSimilar, "cleaned Idx2");
	printGpuArray(negativeIndx, 2*(k-numSimilar), "negativeIdx");

	//Handling dissimilar results
	int* negativeIdxHost = (int*) malloc((2*(k - numSimilar))*sizeof(int));
	cudaMemcpy(negativeIdxHost, negativeIndx, (2*(k - numSimilar))*sizeof(int), cudaMemcpyDeviceToHost);
	handleDissimilar(negativeIdxHost, k-numSimilar);

	k = numSimilar;
	int* rowData;

	cudaMalloc((void**) &rowData, 4 * (dim + 1) * sizeof(int));
	initKernel<<<256, 256>>>(rowData, 0, 4 * (dim + 1));

	int* rowIncrIdx1 = rowData;
	int* rowIncrIdx2 = rowData + (dim + 1);
	int* rowIncr = rowData + 2 * (dim + 1);
	int* prefixRowIncr = rowData + 3 * (dim + 1);

	numThreads = 32;
	numBlocks = 1 + (k / numThreads);
	rowIncrementedKernel<<<numBlocks, numThreads>>>(rowData, cleanedIdx1, cleanedIdx2, (dim + 1), k);

	printGpuArray(rowIncrIdx1, (dim + 1), "idx1 Incr");
	printGpuArray(rowIncrIdx2, (dim + 1), "idx2 Incr");

	arrayAddKernel<<<numBlocks, numThreads>>>(rowIncr, rowIncrIdx1, rowIncrIdx2, (dim + 1));

	dev_ptr_prefix = thrust::device_pointer_cast(rowIncr);
	dev_ptr_prefix_res = thrust::device_pointer_cast(prefixRowIncr);
	thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix + (dim + 1), dev_ptr_prefix_res);

	printGpuArray(rowIncr, (dim + 1), "total increment");
	printGpuArray(prefixRowIncr, (dim + 1), "prefix sum");

	numThreads = 128;
	numBlocks = 1 + (dim / numThreads);
	updateDegreeKernel<<<numBlocks, numThreads>>>(_gpuDegrees, rowIncr, dim);
	numBlocks = 1 + (k / numThreads);
	updateDiagPosKernel<<<numBlocks, numThreads>>>(_gpuDiagPos, cleanedIdx1, cleanedIdx2, k);

	int* newColIdx;
	const int sizeNewColIdx = (getNNZ() + numSimilar * 2);
	cudaMalloc((void**) &newColIdx, sizeNewColIdx * sizeof(int));
	numThreads = 128;
	numBlocks = dim;
	initKernel<<<256, 256>>>(newColIdx, dim + 1, sizeNewColIdx);
	colIdxIncrementKernel<<<numBlocks, numThreads>>>(newColIdx, _gpuColIdx, _gpuRowPtr, prefixRowIncr, dim);

//	printGpuArray(newColIdx, sizeNewColIdx, "new ColIdx: ");

	arrayAddKernel<<<numBlocks, numThreads>>>(prefixRowIncr, prefixRowIncr, _gpuRowPtr, (dim + 1));

//	printGpuArray(prefixRowIncr, (dim + 1), "-> new (future) rowPtr: ");

	int* prefixIndex1;
	cudaMalloc((void**) &prefixIndex1, (dim + 1) * sizeof(int));
	dev_ptr_prefix = thrust::device_pointer_cast(rowIncrIdx1);
	dev_ptr_prefix_res = thrust::device_pointer_cast(prefixIndex1);
	thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix + (dim + 1), dev_ptr_prefix_res);

	numThreads = 32;
	numBlocks = 1 + (dim / numThreads);
	doInsertionKernel<<<numBlocks, numThreads>>>(newColIdx, prefixRowIncr, _gpuRowPtr, cleanedIdx2, prefixIndex1, dim);

//	printGpuArray(newColIdx, sizeNewColIdx, "new ColIdx: ");

	thrust::device_ptr<int> dpIdx1 = thrust::device_pointer_cast(cleanedIdx1);
	thrust::device_ptr<int> dpIdx2 = thrust::device_pointer_cast(cleanedIdx2);
	thrust::sort_by_key(dpIdx2, dpIdx2 + numSimilar, dpIdx1);

	printf("Resorting...\n");
	//printGpuArray(cleanedIdx1, numSimilar, "cleaned Idx1");
	//printGpuArray(cleanedIdx2, numSimilar, "cleaned Idx2");

	arrayAddKernel<<<numBlocks, numThreads>>>(_gpuRowPtr, prefixIndex1, _gpuRowPtr, dim + 1);

//	printGpuArray(_gpuRowPtr, dim + 1, "\t Intermediate old RowPtr");

	dev_ptr_prefix = thrust::device_pointer_cast(rowIncrIdx2);
	thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix + (dim + 1), dev_ptr_prefix_res);

	doInsertionKernel<<<numBlocks, numThreads>>>(newColIdx, prefixRowIncr, _gpuRowPtr, cleanedIdx1, prefixIndex1, dim);

	printGpuArray(newColIdx, sizeNewColIdx, "new ColIdx: ");

	num_similar += numSimilar;
	cudaFree(_gpuColIdx);
	_gpuColIdx = newColIdx;
	cudaFree(_gpuRowPtr);
	_gpuRowPtr = prefixRowIncr;

	cudaFree(rowData);
	cudaFree(rowIncrIdx1);
	cudaFree(rowIncrIdx2);
	cudaFree(cleanedIdx1);
	cudaFree(cleanedIdx2);
	cudaFree(prefixSumResult);
	cudaFree(prefixIndex1);
	cudaFree(negativeIndx);
	free(negativeIdxHost);


	if (true)
	{
		/********* TESTING ****************/
		printf("AFTER UPDATE ---> num similar = %i nnz = %i \n", num_similar, getNNZ());
		printGpuArray(_gpuColIdx, getNNZ(), "colIdx");
		printGpuArray(_gpuRowPtr, dim + 1, "rowPtr");
		printGpuArray(_gpuDiagPos, dim, "diagPos");
		printGpuArray(_gpuDegrees, dim, "degrees");
		getValueArr(true);
		getColumn(3);
		/**********************************/
	}


}

void GPUSparse::set(int i, int j, bool val)
{
	printf("ERROR: set(int, int, bool) not allowed for SPARSE Matrix");
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
	bool verbose = true;

	int nnz = getNNZ();

	float _cpuLambda_times_dim = dim * lambda; //FIXME do not compute each time
	cudaMemcpyToSymbol(lambda_times_dim, &_cpuLambda_times_dim, sizeof(float));

	float* gpuValues;
	cudaMalloc((void**) &gpuValues, nnz * sizeof(float));
	int NUM_THREADS = 512;
	int NUM_BLOCKS = 1;
	scatterKernel<<<NUM_BLOCKS, NUM_THREADS>>>(gpuValues, nnz);

	NUM_THREADS = 512;
	NUM_BLOCKS = 1 + (dim / NUM_THREADS);
	scatterDiagonalKernel<<<NUM_BLOCKS, NUM_THREADS>>>(gpuValues, _gpuRowPtr, _gpuDegrees, _gpuDiagPos, dim);

	if (!gpuPointer)
	{
		float* valuesCPU = (float*) malloc(nnz * sizeof(float));
		cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(gpuValues);;

		return valuesCPU;
	}

	if (gpuPointer && verbose)
	{
		float* valuesCPU = (float*) malloc(nnz * sizeof(float));
		cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(float), cudaMemcpyDeviceToHost);

		printf("Value array: ");
		Tester::printArrayFloat(valuesCPU, nnz);
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
//			printf("-1.0 at %i \n", idx);
		}
	}

	float* _gpuColumn;
	cudaMalloc((void**) &_gpuColumn, dim*sizeof(float));
	int* _gpuNumOnes;
	cudaMalloc((void**) &_gpuNumOnes, sizeof(int));
	cudaMemcpy(_gpuColumn, column, dim*sizeof(float), cudaMemcpyHostToDevice);

	const int numThreads = 512;
	const int numBlocks = 1;
	columnWriteKernel<<<numBlocks, numThreads>>>(_gpuColumn, _gpuColIdx, _gpuRowPtr, _gpuNumOnes, columnIdx, dim);

	//test printing
	if(true)
	{
		cudaMemcpy(column, _gpuColumn, dim*sizeof(float), cudaMemcpyDeviceToHost);
		printf("Column %i :", columnIdx);
		Tester::printArrayFloat(column, dim);
	}

	return column;
}

char* GPUSparse::getMatrAsArray()
{
	printf("ERROR: getMatrAsArray() not supported by SPARSE Matrix (return NULL)\n");
	return NULL;
}

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
	printGpuArray(_gpuColIdx, getNNZ(), "colIdx (on GPU)");
	printGpuArray(_gpuRowPtr, dim+1, "rowPtr (on GPU)");
}

void GPUSparse::writeGML(char * filename, bool similar, bool dissimilar,
		bool potential)
{
	//TODO
}

int* GPUSparse::getRowPtr() const
{
	return NULL;
}

int* GPUSparse::getRowPtrDevice() const
{
	return _gpuRowPtr;
}

int* GPUSparse::getColIdx() const
{
	return NULL;
}

int* GPUSparse::getColIdxDevice() const
{
	return _gpuColIdx;
}

unsigned int GPUSparse::getNNZ() const
{
	return num_similar * 2 + nnz_rows;
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
