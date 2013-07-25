/*
 * GPUSparse.cpp
 *
 *  Created on: Jun 12, 2013
 *      Author: Armin Gufler
 */

#include "GPUSparse.h"
#include <stdio.h>
#include <stdlib.h>
#include "Tester.h"
#include "Helper.h"
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <fstream>

#define CHECK_FOR_CUDA_ERROR 0

#define CUDA_CHECK_ERROR() {							\
cudaError_t err = cudaGetLastError();					\
if (cudaSuccess != err) {						\
    fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
            __FILE__, __LINE__, cudaGetErrorString(err) );	\
    exit(EXIT_FAILURE);						\
}									\
}

__device__ __constant__ double lambda_times_dim;

/***************************************************************************************/
/**************** KERNELS TO GET THE VALUE ARRAY ***************************************/
__global__ void scatterKernelDouble(double* dst, int num)
{
	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	for(int pos = i; pos < num; pos += blockDim.x)
	{
		dst[pos] = (-1)*lambda_times_dim;
	}
	//TODO split into a known number of blocks, let each thread write one number

}

__global__ void scatterDiagonalKernelDouble(double* gpuValues, int* gpuRowPtr,
		int* gpuDegrees, int* gpuDiagPos, int dim)
{
	//number of this thread
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < dim)
	{
		int row = i;
		const int valueIndex = gpuRowPtr[row] + gpuDiagPos[row];
		const double valToWrite = 1 + (lambda_times_dim * gpuDegrees[row]);
		gpuValues[valueIndex] = valToWrite;
	}
}
/***************************************************************************************/
/***************************************************************************************/

__global__ void colIdxIncrementKernel(int* colIdx, int* oldColIdx, int* rowPtr, int* incr, int size)
{
	//TODO can this kernel make profitable use of shared memory?

	const unsigned int row = blockIdx.x * gridDim.x + blockIdx.y;

	if (row < size)
	{
		const unsigned int rowIdx = threadIdx.x;

		for (int j = rowIdx; j < (rowPtr[row + 1] - rowPtr[row]); j += blockDim.x)
		{
			colIdx[rowPtr[row] + j + incr[row]] = oldColIdx[rowPtr[row] + j];
		}
		//Note: a single thread has to do more than one step only if blockDim.x < length of row
		// Number of blocks has to be >= number of rows for this kernel!
	}


}

__global__ void arrayAddKernel(int* res, int* _a1, int* _a2, int size)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < size)
	{
		res[i] = _a1[i] + _a2[i];
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

__global__ void columnWriteKernelDouble(double* dst, int* colIdx, int* rowPtr, int* numOnes, int col, int dim)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

	if(tIdx == 0){
		//put into 'shared' memory the amount of 1's of this column
		numOnes[0] = rowPtr[col+1] - rowPtr[col];
	}
	__syncthreads();

	for(int i = tIdx; i < numOnes[0] && i < dim; i+=blockDim.x)
	{
		const int idx_in_colIdx = rowPtr[col]+i;
		dst[colIdx[idx_in_colIdx]] = 1.0;
	}

	__syncthreads();
	if(tIdx == 0)
	{
		//diagonal to zero
		dst[col] = 0.0;
	}
}

__global__ void setupCurandkernel(curandState *state, int seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    /* (from CUDA CURAND documentation) */
    curand_init(seed, tid, 0, &state[tid]);
}


//kernel for randomly filling index arrays with not yet compared images
//TODO can for sure be implemented in a much more elegant way ;)
__global__ void randomComparisonFillKernel(int* idx1, int* idx2, int* rowPtr,
							int* colIdx, const int k, const int dim, curandState* state)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

	if(tIdx < k)
	{
		int myRandomRow;
		int rangeSize = dim/k;
		const int myLowLimit = tIdx * rangeSize; //rand must be >= this limit
		const int myUpperLimit = myLowLimit + rangeSize - 1; //rand must be < this limit


		curandState localState = state[tIdx];
		if(myLowLimit == myUpperLimit)
		{
			myRandomRow = myLowLimit;
		}
		else
		{
			float uniRand = curand_uniform(&localState);
			state[tIdx] = localState; //back to global

			myRandomRow = myLowLimit + (uniRand * (myUpperLimit - myLowLimit)) - 1; // should be in [low,up)
		}

		idx1[tIdx] = myRandomRow;

		int currentlyFound = dim+1;
		int last = -1;
		const int minElem = myRandomRow+1; //only lower diagonal matrix allowed
		for(int j = rowPtr[myRandomRow]; j < rowPtr[myRandomRow+1]; j++)
		{
			if(colIdx[j] != last+1)
			{
				//an empty spot found

				//check if not diagonal element (i,i) and >= min
				if(last+1 != myRandomRow && last+1 >= minElem){
					currentlyFound = last+1;
				}

			}
			last = colIdx[j];
		}
		
		//with high probability run on... at larger elments
		if(((curand_uniform(&localState) < .80) && (last < dim-1)) || ((last < dim-1) && currentlyFound==dim+1))
		{
		float diff = dim - minElem;
		float breakProb = 1/ diff;
			for(int j = last; j < dim; j++)
			{
				if(j != myRandomRow && j >= minElem)
				{
					currentlyFound = j;
					if(curand_uniform(&localState) < breakProb)
					  break;
					breakProb += 1/diff;
				}
			}
		}
		idx2[tIdx] = currentlyFound;
	}
}

//just checks if no values > dim+1 or < 0 are in given array
__global__ void checkRandoms(int* idx1, const int k, const int dim, int* errors)
{
	int tIdx = threadIdx.x + blockDim.x * blockIdx.x;

	if(tIdx < k)
	{
		if(idx1[tIdx] > dim+1 || idx1[tIdx] < 0)
		{
			idx1[tIdx] = dim+1;
			atomicAdd(&errors[0], 1);
		}
	}
}

GPUSparse::GPUSparse(unsigned int _dim, float _lambda) :
		dim(_dim), lambda(_lambda)
{
	num_similar = 0;

	cudaMalloc((void**) &_gpuRowPtr, (dim+1)*sizeof(int));
	cudaMalloc((void**) &_gpuColIdx, dim*sizeof(int));
	cudaMalloc((void**) &_gpuDegrees, dim*sizeof(int));
	cudaMalloc((void**) &_gpuDiagPos, dim*sizeof(int));
	cudaMalloc((void**) &_gpuColumnPtr, dim*sizeof(double));

	const int numThreads = 128;
	const int numBlocks = 1 + (dim/numThreads);
	initColIdxRowPtrKernel<<<numBlocks, numThreads>>>(_gpuColIdx, _gpuRowPtr, dim);
	initKernel<<<numBlocks, numThreads>>>(_gpuDegrees, 0, dim);
	initKernel<<<numBlocks, numThreads>>>(_gpuDiagPos, 0, dim);
	
	lastRandomK = -1;
}

GPUSparse::~GPUSparse()
{
	cudaFree(_gpuColIdx);
	cudaFree(_gpuRowPtr);
	cudaFree(_gpuDegrees);
	cudaFree(_gpuDiagPos);
	cudaFree(_gpuColumnPtr);
}

//TODO host thread ?!
void GPUSparse::handleDissimilar(int* idxData, int num)
{
	for(int i = 0; i < num*2; i+=2)
	{
		int col = idxData[i];
		int row = idxData[i+1];

		if(col > dim) //not allowed (safety check)
			break;
		if(col == dim) //avoid diagonal
			continue;

		addDissimilarToColumn(col, row);
		addDissimilarToColumn(row, col);
	}
}

void GPUSparse::updateSparseStatus(int* _idx1, int* _idx2, int* _res, int _k)
{
	const bool verbose = false;

	//this conversion is done just for testing convenience
	int* idx1 = _idx1;
	int* idx2 = _idx2;
	int* res = _res;
	int k = _k;

	//get number of non-zero elements before update
	const int nnzBeforeUpdate = getNNZ();

	if (verbose)
	{
		print();
		printf("New Input for update:\n");
		Helper::printGpuArray(idx1, k, "idx1");
		Helper::printGpuArray(idx2, k, "idx2");
		Helper::printGpuArray(res, k, "res ");
		printf("\n");
	}

	//used for kernel configuration
	int numThreads;
	int numBlocks;

	int* prefixSumResult;
	cudaMalloc((void**) &prefixSumResult, (k) * sizeof(int));
	thrust::device_ptr<int> dev_ptr_prefix_res = thrust::device_pointer_cast(prefixSumResult);
	thrust::device_ptr<int> dev_ptr_prefix = thrust::device_pointer_cast(res);
	thrust::exclusive_scan(dev_ptr_prefix, dev_ptr_prefix + k, dev_ptr_prefix_res);
	
#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif	
	
	//TODO move numSimilar computation to GPU, directly in kernel with argument?
	int numSimilar;
	int lastResVal;
	cudaMemcpy(&numSimilar, prefixSumResult + (k-1), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&lastResVal, res + (k-1), sizeof(int), cudaMemcpyDeviceToHost);

	if(lastResVal == 1)
		numSimilar++;

//	printf("numSimilar = %i \n", numSimilar);

	int* cleanedIdx;
	int* negativeIndx;
	cudaMalloc((void**) &cleanedIdx, numSimilar * 2 * sizeof(int));
	cudaMalloc((void**) &negativeIndx, (2*(k - numSimilar))*sizeof(int));
	int* cleanedIdx1 = cleanedIdx;
	int* cleanedIdx2 = cleanedIdx + numSimilar;
	
	numThreads = 32;
	numBlocks = 1 + (k / numThreads);
	cleanIndexArrays<<<numBlocks, numThreads>>>(cleanedIdx1, cleanedIdx2, negativeIndx, idx1, idx2, res, prefixSumResult, k);

	cudaFree(prefixSumResult);

	//Handling dissimilar results
	int* negativeIdxHost = (int*) malloc((2*(k - numSimilar))*sizeof(int));
	cudaMemcpy(negativeIdxHost, negativeIndx, (2*(k - numSimilar))*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(negativeIndx);

	handleDissimilar(negativeIdxHost, k-numSimilar);
	

	k = numSimilar;
	int* rowData;

	cudaMalloc((void**) &rowData, 4 * (dim + 1) * sizeof(int));
	initKernel<<<512, 256>>>(rowData, 0, 4 * (dim + 1));

	int* rowIncrIdx1 = rowData;
	int* rowIncrIdx2 = rowData + (dim + 1);
	int* rowIncr = rowData + 2 * (dim + 1);
	int* prefixIndex1 = rowData + 3 * (dim + 1);

	int* prefixRowIncr; //will be new rowPtr
	cudaMalloc((void**) &prefixRowIncr, (dim + 1) * sizeof(int));

	numThreads = 32;
	numBlocks = 1 + (k / numThreads);
	rowIncrementedKernel<<<numBlocks, numThreads>>>(rowData, cleanedIdx1, cleanedIdx2, (dim + 1), k);

	numThreads = 256;
	numBlocks = 1 + ((dim+1)/numThreads);
	arrayAddKernel<<<numBlocks, numThreads>>>(rowIncr, rowIncrIdx1, rowIncrIdx2, (dim + 1));

	dev_ptr_prefix = thrust::device_pointer_cast(rowIncr);
	dev_ptr_prefix_res = thrust::device_pointer_cast(prefixRowIncr);
	thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix + (dim + 1), dev_ptr_prefix_res);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	numThreads = 128;
	numBlocks = 1 + (dim / numThreads);
	updateDegreeKernel<<<numBlocks, numThreads>>>(_gpuDegrees, rowIncr, dim);
	numBlocks = 1 + (k / numThreads);
	updateDiagPosKernel<<<numBlocks, numThreads>>>(_gpuDiagPos, cleanedIdx1, cleanedIdx2, k);

	int* newColIdx;
	const int sizeNewColIdx = (getNNZ() + numSimilar * 2);
	cudaMalloc((void**) &newColIdx, sizeNewColIdx * sizeof(int));

	initKernel<<<512, 256>>>(newColIdx, dim + 1, sizeNewColIdx);

	numThreads = 128;
	numBlocks = dim;
	int gridDim = 1 + sqrt(dim);
	dim3 blockGrid(gridDim,gridDim);
	colIdxIncrementKernel<<<blockGrid, numThreads>>>(newColIdx, _gpuColIdx, _gpuRowPtr, prefixRowIncr, dim);


	numThreads = 256;
	numBlocks = 1 + ((dim+1)/numThreads);
	arrayAddKernel<<<numBlocks, numThreads>>>(prefixRowIncr, prefixRowIncr, _gpuRowPtr, (dim + 1));
	

	dev_ptr_prefix = thrust::device_pointer_cast(rowIncrIdx1);
	dev_ptr_prefix_res = thrust::device_pointer_cast(prefixIndex1);
	thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix + (dim + 1), dev_ptr_prefix_res);
	
	numThreads = 32;
	numBlocks = 1 + (dim / numThreads);
	doInsertionKernel<<<numBlocks, numThreads>>>(newColIdx, prefixRowIncr, _gpuRowPtr, cleanedIdx2, prefixIndex1, dim);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	//resorting such that sorted after idx2 array
	thrust::device_ptr<int> dpIdx1 = thrust::device_pointer_cast(cleanedIdx1);
	thrust::device_ptr<int> dpIdx2 = thrust::device_pointer_cast(cleanedIdx2);
	thrust::sort_by_key(dpIdx2, dpIdx2 + numSimilar, dpIdx1);

	arrayAddKernel<<<numBlocks, numThreads>>>(_gpuRowPtr, prefixIndex1, _gpuRowPtr, dim + 1);

	dev_ptr_prefix = thrust::device_pointer_cast(rowIncrIdx2);
	thrust::inclusive_scan(dev_ptr_prefix, dev_ptr_prefix + (dim + 1), dev_ptr_prefix_res);

	//insertion sort of new elements on column index
	doInsertionKernel<<<numBlocks, numThreads>>>(newColIdx, prefixRowIncr, _gpuRowPtr, cleanedIdx1, prefixIndex1, dim);

	num_similar += numSimilar;


//	Tester::testCSRMatrixUpdate(Helper::downloadGPUArrayInt(originalRowPtr, dim+1), Helper::downloadGPUArrayInt(_gpuColIdx, oldNNZ), Helper::downloadGPUArrayInt(_gpuDegrees, dim),
//								Helper::downloadGPUArrayInt(prefixRowIncr, dim+1), Helper::downloadGPUArrayInt(newColIdx, getNNZ()),
//								Helper::downloadGPUArrayInt(idx1, _k), Helper::downloadGPUArrayInt(idx2,_k), Helper::downloadGPUArrayInt(res,_k), dissimilarMap, dim, _k);
	//cudaFree(originalRowPtr);

	cudaFree(_gpuColIdx);
	_gpuColIdx = newColIdx;
	cudaFree(_gpuRowPtr);
	_gpuRowPtr = prefixRowIncr;

	cudaFree(rowData);
	cudaFree(cleanedIdx);
	
	free(negativeIdxHost);

	if (verbose)
	{
		/********* TESTING ****************/
		printf("AFTER UPDATE ---> num similar = %i nnz = %i \n", num_similar, getNNZ());
		Helper::printGpuArray(_gpuColIdx, getNNZ(), "colIdx");
		Helper::printGpuArray(_gpuRowPtr, dim + 1, "rowPtr");
		Helper::printGpuArray(_gpuDiagPos, dim, "diagPos");
		Helper::printGpuArray(_gpuDegrees, dim, "degrees");
		//getValueArr(true);
		//getColumn(3);
		/**********************************/
	}

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif
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

double* GPUSparse::getValueArrayDouble(bool gpuPointer) const
{
	bool verbose = false;

	int nnz = getNNZ();

	double _cpuLambda_times_dim_double = dim * lambda; //FIXME do not compute each time
	cudaMemcpyToSymbol(lambda_times_dim, &_cpuLambda_times_dim_double, sizeof(double));

	double* gpuValues;
	cudaMalloc((void**) &gpuValues, nnz * sizeof(double));
	int NUM_THREADS = 512;
	int NUM_BLOCKS = 1;
	scatterKernelDouble<<<NUM_BLOCKS, NUM_THREADS>>>(gpuValues, nnz);

	NUM_THREADS = 512;
	NUM_BLOCKS = 1 + (dim / NUM_THREADS);
	scatterDiagonalKernelDouble<<<NUM_BLOCKS, NUM_THREADS>>>(gpuValues, _gpuRowPtr, _gpuDegrees, _gpuDiagPos, dim);

	if (!gpuPointer)
	{
		double* valuesCPU = (double*) malloc(nnz * sizeof(double));
		cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(gpuValues);;

		return valuesCPU;
	}

	if (gpuPointer && verbose)
	{
		double* valuesCPU = (double*) malloc(nnz * sizeof(double));
		cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(double), cudaMemcpyDeviceToHost);

		printf("Value array: ");
		Tester::printArray(valuesCPU, nnz);
	}

//	double* valuesCPU = (double*) malloc(nnz * sizeof(double));
//				cudaMemcpy(valuesCPU, gpuValues, nnz * sizeof(double), cudaMemcpyDeviceToHost);
//		Tester::testValueArray(Helper::downloadGPUArrayInt(_gpuRowPtr, dim+1), Helper::downloadGPUArrayInt(_gpuColIdx, nnz), Helper::downloadGPUArrayInt(_gpuDegrees, dim),
//							dim, nnz, lambda, valuesCPU);

	return gpuValues;
}


double* GPUSparse::getColumnDouble(int columnIdx) const
{
	double* column = new double[dim];
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

	int* _gpuNumOnes;
	cudaMalloc((void**) &_gpuNumOnes, sizeof(int));
	cudaMemcpy(_gpuColumnPtr, column, dim*sizeof(double), cudaMemcpyHostToDevice);

	const int numThreads = 512;
	const int numBlocks = 1;
	columnWriteKernelDouble<<<numBlocks, numThreads>>>(_gpuColumnPtr, _gpuColIdx, _gpuRowPtr, _gpuNumOnes, columnIdx, dim);

	//testing
	if(false)
	{
		cudaMemcpy(column, _gpuColumnPtr, dim*sizeof(double), cudaMemcpyDeviceToHost);
		Tester::testColumn(dissimilarMap, Helper::downloadGPUArrayInt(_gpuRowPtr, dim+1), Helper::downloadGPUArrayInt(_gpuColIdx, getNNZ()),
					columnIdx, dim, column);
	}

	cudaFree(_gpuNumOnes);
	delete[] column;

	return _gpuColumnPtr;
}

char* GPUSparse::getMatrAsArray()
{
	printf("ERROR: getMatrAsArray() not supported by SPARSE Matrix (return NULL)\n");
	return NULL;
}

char GPUSparse::getVal(int i, int j)
{
	printf("ERROR: getVal(int, int) to get a single value not supported by SPARSE Matrix \n");
	return -1;
}

int GPUSparse::getSimilarities()
{
	return getNNZ() - dim;
}

void GPUSparse::print()
{
	printf("###INFO about SPARSE MATRIX###\n");
	printf("dim = %i, lambda = %f, nnz = %i \n", dim, lambda, getNNZ());
//	Helper::printGpuArray(_gpuColIdx, getNNZ(), "colIdx (on GPU)");
//	Helper::printGpuArray(_gpuRowPtr, dim+1, "rowPtr (on GPU)");
}

void GPUSparse::writeGML(char * filename, bool similar, bool dissimilar,
		bool potential)
{
	//TODO
}

int* GPUSparse::getRowPtrDevice() const
{
	return _gpuRowPtr;
}

int* GPUSparse::getColIdxDevice() const
{
	return _gpuColIdx;
}

unsigned int GPUSparse::getNNZ() const
{
	return num_similar * 2 + dim;
}


//TODO not complete yet.
void GPUSparse::fillRandomCompareIndices(int* idx1, int* idx2, int* res, const int k)
{
//	Helper::printGpuArray(_gpuColIdx, getNNZ(), "colIdx (on GPU)");
//	Helper::printGpuArray(_gpuRowPtr, dim+1, "rowPtr (on GPU)");
	
	const int numThreads = 256;
	const int numBlocks = 1 + (numThreads/k);

	if(lastRandomK != k)
	{
	  if(lastRandomK != -1)
	  {
		cudaFree(devStates);
	  }

	  const int cudaInitSeed = rand() % 1234567;
	  
	  cudaMalloc((void **)&devStates, numThreads * numBlocks *  sizeof(curandState));
	  setupCurandkernel<<<numBlocks, numThreads>>>(devStates, cudaInitSeed);
	}
	
	lastRandomK = k;

	initKernel<<<256, 256>>>(res, 0, k); //TODO necessary to initialize res ?

	initKernel<<<256, 256>>>(idx1, dim+1, k);

	initKernel<<<256, 256>>>(idx2, dim+1, k);

	cudaDeviceSynchronize();
	
	randomComparisonFillKernel<<<numBlocks, numThreads>>>(idx1, idx2, _gpuRowPtr, _gpuColIdx, k, dim, devStates);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	int* h_idx1 = Helper::downloadGPUArrayInt(idx1, k);
	int* h_idx2 = Helper::downloadGPUArrayInt(idx2, k);
	
	for(int i = 0; i < k; i++)
	{
	  myElemMap::const_iterator it = dissimilarMap.find(h_idx1[i]);
	  if (it != dissimilarMap.end())
	  {
		if(it->second.find(h_idx2[i]) != it->second.end())
		{
		  h_idx1[i] = dim+1;
		  h_idx2[i] = dim+1;
		}
	  }
	}
	
	cudaMemcpy(idx1, h_idx1, k*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(idx2, h_idx2, k*sizeof(int), cudaMemcpyHostToDevice);
	
	
	/* ERROR checking
	int* errors;
	cudaMalloc((void**) &errors, sizeof(int));
	cudaMemset(errors, 0, sizeof(int));
	checkRandoms<<<numBlocks, numThreads>>>(idx1, k, dim, errors);
	int h_errors;
	cudaMemcpy(&h_errors, errors, sizeof(int),  cudaMemcpyDeviceToHost);
	if(h_errors != 0){
		printf("Error on random fill [idx1]! Aborting...");
		exit(EXIT_FAILURE);
	}
	cudaMemset(errors, 0, sizeof(int));
	checkRandoms<<<numBlocks, numThreads>>>(idx1, k, dim, errors);
	cudaMemcpy(&h_errors, errors, sizeof(int),  cudaMemcpyDeviceToHost);
	if(h_errors != 0){
		printf("Error on random fill [idx2]! Aborting...");
		exit(EXIT_FAILURE);
	}
	*/

	thrust::device_ptr<int> dpIdx1 = thrust::device_pointer_cast(idx1);
	thrust::device_ptr<int> dpIdx2 = thrust::device_pointer_cast(idx2);
	thrust::sort_by_key(dpIdx1, dpIdx1 + k, dpIdx2);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	//free stuff
	free(h_idx1);
	free(h_idx2);

//	Helper::printGpuArray(idx1, k, "i:");
//	Helper::printGpuArray(idx2, k, "j:");
}

void GPUSparse::logSimilarToFile(const char *path, ImageHandler* iHandler) const
{
	std::ofstream file;
	file.open(path);

	file << iHandler->getDirectoryPath() << "\n";

	const char* f1;
	const char* f2;

	for(myElemMap::const_iterator it = dissimilarMap.begin();
			it != dissimilarMap.end(); it++)
	{
		const int column = it->first;
		f1 = iHandler->getImage(column);
		std::set<int> dis = it->second;
		for (std::set<int>::const_iterator lIter = dis.begin(); lIter != dis.end(); ++lIter)
		{
			f2 = iHandler->getImage(*lIter);
			file << f1 << "\t" << f2 << "\t-1\n";
		}
	}

	int* h_rowPtr = Helper::downloadGPUArrayInt(_gpuRowPtr, dim+1);
	int* h_colIdx = Helper::downloadGPUArrayInt(_gpuColIdx, getNNZ());

	for(int i = 0; i < dim; i++)
	{
		const int start = h_rowPtr[i];
		const int end = h_rowPtr[i+1];
		f2 = iHandler->getImage(i);
		for(int idx = start; idx < end; idx++)
		{
			const int column = h_colIdx[idx];
			if(column == i)
				continue;
			f1 = iHandler->getImage(column);
			file << f1 << "\t" << f2 << "\t1\n";
		}
	}

	free(h_rowPtr);
	free(h_colIdx);

	file.close();
}

