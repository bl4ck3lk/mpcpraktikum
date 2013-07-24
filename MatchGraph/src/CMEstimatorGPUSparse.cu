/*
 * CMEstimatorGPUSparse.cu
 *
 * Generates a list of indices containing the i, j index of approx. the 
 * k-best confidence measure values. 
 * The List of indices is generated column wise after Cula solved the
 * linear equation system. This class uses the already stored memory
 * of the equation solver and extracts the k best values of the specific
 * column.
 *
 *  Created on: 19.06.2013
 *      Author: Fabian, Armin
 */

#include "GPUSparse.h"
#include "CMEstimatorGPUSparse.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <ctime>
#include <vector>
#include <algorithm> /* std::find */
#include <stdio.h> /* printf */
#include <float.h> /* FLT_MAX */
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cula_sparse.h>
#include "Tester.h"
#include "Helper.h"

#define CHECK_FOR_CUDA_ERROR 0

#define CUDA_CHECK_ERROR() {							\
    cudaError_t err = cudaGetLastError();					\
    if (cudaSuccess != err) {						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
}

inline __int64_t continuousTimeNs()
 {
         timespec now;
         clock_gettime(CLOCK_REALTIME, &now);

         __int64_t result = (__int64_t ) now.tv_sec * 1000000000
                         + (__int64_t ) now.tv_nsec;

         return result;
 }

const int THREADS = 128;

__device__ int d_notWritten;

//Initialize index arrays
static __global__ void initIndexArrays(int* d_idx1, int* d_idx2, int* d_res, int size, unsigned int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		d_idx1[idx] = dim+1;
		d_idx2[idx] = dim+1;
		d_res[idx] = 0;
	}
}


//Initialize indices
static __global__ void initKernel(long* gpuIndices, double* x, double* b, const int dim, const int columnIdx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < dim)
	{
		if (columnIdx >= idx || 0 != b[idx]) //diagonal element or known element or upper diagonal matrix element
		{
			gpuIndices[idx] = -1;
			//assign very low value to avoid them getting chosen
			x[idx] = -DBL_MAX;
		}
		else
		{
			//assign index value based on the overall matrix dimension (continuous idx)
			gpuIndices[idx] = columnIdx + idx * dim;
		}
	}
}

//Write indices to respective index arrays
static __global__ void saveIndicesKernel(long* gpuIndices, int* d_idx1, int* d_idx2, int dim, int kBest, int kBestForThisColumn, int currIndexNr)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int write_idx = t_idx + currIndexNr;

	if (t_idx < dim && write_idx < kBest && t_idx < kBestForThisColumn)
	{
		int gpuIdx = gpuIndices[t_idx]; //size(gpuIndices) > kBestForThisColumn > t_idx

		if (-1 != gpuIdx)
		{
			int i = gpuIdx/dim;
			int j = gpuIdx%dim;

			d_idx1[write_idx] = i;
			d_idx2[write_idx] = j;
		}
		else
		{
			atomicAdd(&d_notWritten, 1); //save nr of threads not writing anything
		}
	}
}

CMEstimatorGPUSparse::CMEstimatorGPUSparse() {
	lastSize = 0;
	d_idx1 = NULL;
	d_idx2 = NULL;
	d_res = NULL;

	totalTime = 0;

	/* cula initialization */
	//initCula();
	printf("Using random estimator (CMEstimatorGPUSparse)\n");
}


CMEstimatorGPUSparse::~CMEstimatorGPUSparse() {
	//free device pointer
	if (d_idx1 != NULL) cudaFree(d_idx1);
	if (d_idx2 != NULL) cudaFree(d_idx2);
	if (d_res != NULL) cudaFree(d_res);
	printf("Total solver time: %f\n", totalTime*(1/(double)1000000000));
}

int* CMEstimatorGPUSparse::getIdx1Ptr()
{
	return d_idx1;
}

int* CMEstimatorGPUSparse::getIdx2Ptr()
{
	return d_idx2;
}

int* CMEstimatorGPUSparse::getResPtr()
{
	return d_res;
}

void CMEstimatorGPUSparse::initCula()
{
//	//config solver
//	config.relativeTolerance = 1e-6;
//	config.maxIterations = 300;
//	config.maxRuntime = 10;
//
//	culaSparseCreate(&handle); //create library handle
//	culaSparseConfigInit(handle, &config); //initialize values
//	culaSparseCreatePlan(handle, &plan); //create execution plan
//	culaSparseSetCudaDevicePlatform(handle, plan, 0); //use the CUDA-device platform (interprets given pointer as device pointers)
//	culaSparseSetCgSolver(handle, plan, 0); //associate CG solver with the plan
//	culaSparseSetJacobiPreconditioner(handle, plan, 0); //associate jacobi preconditioner with the plan
}

//Allocate device memory for index pointers and clear last used pointers
//(for dynamic change of kBest values index-arays)
void CMEstimatorGPUSparse::initIdxDevicePointers(int size, unsigned int dim)
{
	//clear previous pointers
	if (d_idx1 != NULL) cudaFree(d_idx1);
	if (d_idx2 != NULL) cudaFree(d_idx2);
	if (d_res != NULL) cudaFree(d_res);

	//allocate new device memory
	cudaMalloc((void**) &d_idx1, size * sizeof(int));
	cudaMalloc((void**) &d_idx2, size * sizeof(int));
	cudaMalloc((void**) &d_res, size * sizeof(int));

	//Kernel settings for index array
	int numBlocks = (size + THREADS - 1) / THREADS;
	initIndexArrays<<<numBlocks, THREADS>>>(d_idx1, d_idx2, d_res, size, dim);

//	printf("[ESTIMATOR]: Device index arrays with size %i allocated.\n",size);
}

/*
 * Determines column-wise the best confidence measures of the specific column and saves it indices in two arrays.
 * Returns the number of actually written slots.
 */
int CMEstimatorGPUSparse::determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, int dim, int kBest, int kBestForThisColumn, int currIndexNr)
{
	//Allocate index array on GPU
	long* gpuIndices;
	cudaMalloc((void**) &gpuIndices, dim * sizeof(long));

	//wrap raw pointer with device pointer
	thrust::device_ptr<long> dp_gpuIndices = thrust::device_pointer_cast(gpuIndices);

	//Kernel settings for index array
	int numBlocks = (dim + THREADS - 1) / THREADS;
	int numThreads = THREADS;

	/* Init indices array such that indices = [-1,1,2,-1,...,dim-1], whereas the respective
	 * diagonal element is -1 as well as elements that are already compared or within the upper
	 * diagonal matrix.
	 * For already known elements (i.e. bColumnDevice[i] != 0), xColumnDevice[i] will be
	 * assigned a very low value to prevent them from getting chosen later.
	 */
	initKernel<<<numBlocks, numThreads>>>(gpuIndices, xColumnDevice, bColumnDevice, dim, columnIdx);
#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	//wrap column device pointer
	thrust::device_ptr<double> dp_xColumn = thrust::device_pointer_cast(xColumnDevice);

	//sort x column and index array respectively
	//already known values will be the last ones due to initialization
	thrust::sort_by_key(dp_xColumn, dp_xColumn + dim, dp_gpuIndices, thrust::greater<double>());

//	Helper::printGpuArrayD(xColumnDevice, 10, "Top10");

	//maybe recast pointers? (from thrust)
	//unsigned int * raw_ptr = thrust::raw_pointer_cast(dev_data_ptr);

	//save 'kBestForThisColumn' indices if possible (maybe not enough indices available)
	numBlocks = (kBest + THREADS - 1) / THREADS;
	int notWritten = 0;
	cudaMemcpyToSymbol(d_notWritten, &notWritten, sizeof(int));
	saveIndicesKernel<<<numBlocks, numThreads>>>(gpuIndices, d_idx1, d_idx2, dim, kBest, kBestForThisColumn, currIndexNr);
	cudaMemcpyFromSymbol(&notWritten, d_notWritten, sizeof(int));

//	printf("notWritten = %i\n", notWritten);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	//free memory
	cudaFree(gpuIndices);

	return kBestForThisColumn - notWritten;
}

void CMEstimatorGPUSparse::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
//	printf("[ESTIMATOR]: Determine kBest confidence measures on GPU (column-wise).\n");

	//invoked only on sparse MatrixHandler
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);
	unsigned int dim = T_sparse->getDimension();

	//indices cache

	/* if index array size changed since last use, allocate new device memory
	 * with new size and free old device memory. Otherwise reuse device memory.
	 */
	if (kBest != lastSize)
	{
		initIdxDevicePointers(kBest, dim);
		lastSize = kBest;
	}
	int countIndices = 0;

	//set up data for solver
	unsigned int nnz = T_sparse->getNNZ();
	double* d_values = T_sparse->getValueArrayDouble(true);
	int* d_colIdx = T_sparse->getColIdxDevice();
	int* d_rowPtr = T_sparse->getRowPtrDevice();

	//x-vector
	double* d_x;
	cudaMalloc((void**) &d_x, dim * sizeof(double));

//	Helper::printGpuArrayD(d_values, nnz, "Values");

	//b-vector
	double* d_b;

	//Reinitialize cula to ensure proper execution
	//config solver
	culaSparseHandle handle;
	culaSparsePlan plan;
	culaSparseConfig config;

	culaSparseCreate(&handle); //create library handle
	culaSparseCreatePlan(handle, &plan); //create execution plan

    culaSparseCudaDeviceOptions platformOpts;
    //use the CUDA-device platform (interprets given pointer as device pointers)
    culaSparseStatus statCula = culaSparseCudaDeviceOptionsInit(handle, &platformOpts);
	platformOpts.deviceId = 0;
	platformOpts.debug = 0;
	statCula = culaSparseSetCudaDevicePlatform(handle, plan, &platformOpts);

	culaSparseConfigInit(handle, &config); //initialize config values
//	config.relativeTolerance = 1e-4;
//	config.maxIterations = 50;
//	config.maxRuntime = 1;
//	config.useBestAnswer = 1;

	culaSparseSetCgSolver(handle, plan, 0); //associate CG solver with the plan
	culaSparseSetJacobiPreconditioner(handle, plan, 0); //associate jacobi preconditioner with the plan

	int noError = 0;
	int solverTrials = 0;
	int determinedIndicesByNow = 0;
	int column = rand() % dim;
	int xBestForThisColumn = kBest*0.001;
	char* colsVisited = (char*)malloc(sizeof(char)*dim);
	memset(colsVisited, 0, dim);
	//printf("[CMESTIMATOR]: Solve Eq. system column by column.\n");
	__int64_t startCula = continuousTimeNs();
	for(int i = 0; i < (dim) && countIndices < kBest; i++) //if enough values are gathered, stop computation
	{
		//0. determine number of best values for this column
		//The bigger i, the less best indices are determined for this column
//		int xBestForThisColumn = ((dim-i)/(0.5*dim*(dim-1))) * kBest;
		if (!xBestForThisColumn) xBestForThisColumn = 1; //at least 1 per column
		//take into account that probably not as many indices as needed can be determined, so try to get them in the next column
		int determineXforThisColumn = xBestForThisColumn + (determinedIndicesByNow - countIndices);
		//1. Compute confidence measure for this column (solve Ax=b)

		d_b = T_sparse->getColumnDouble(column);
		//Helper::printGpuArrayD(d_b, dim, "B");
		colsVisited[column] = 1;

		culaSparseStatus res = computeConfidenceMeasure(handle, plan, config, dim, nnz, d_values, d_rowPtr, d_colIdx, d_x, d_b);

		solverTrials++;

		if(res == culaSparseUnspecifiedError || res == culaSparseRuntimeError || res == culaSparseInteralError)
		{
			//A bad CULA error occurred

			if(res == culaSparseRuntimeError)
			{
				printf("Exiting due to CULA Runtime Error\n");
			}
			else
			{
				printf("Exiting due to CULA internal or unspecified ERROR!\n");
			}

			T_sparse->print();
			exit(EXIT_FAILURE);
		}
		else
		{
			if(res == culaSparseNoError)
				noError++;

//			Helper::printGpuArrayD(d_x, dim, "X");

			//2. get indices of x best confidence measure values
			int writtenIndices = determineBestConfMeasures(d_x, d_b, column, dim, kBest, determineXforThisColumn, countIndices);
			countIndices += writtenIndices;

		}

		//cudaFree(d_b);

		determinedIndicesByNow += xBestForThisColumn; // #indices that should have been determined

		column = rand() % dim;
		const int startCol = column;
		while(colsVisited[column] == 1)
		{
			column = (column + 1) % dim;
			if(column == startCol)
				break;
		}

//		printf("Column %i, try to determine %i best values. Actually determined by now %i values\n", i, determineXforThisColumn, countIndices);
	}
	__int64_t solverDiff = continuousTimeNs()-startCula;
	totalTime += solverDiff;
	//printf("Solver time [%i of %i NO ERROR]: %f\n", noError, solverTrials, solverDiff*(1/(double)1000000000));
	printf("%f\t%i\t", solverDiff*(1/(double)1000000000), (solverTrials-noError));
	free(colsVisited);

	//printf("After solving [%i of %i NO ERROR]! Going to sort with thrust\n", noError, solverTrials);
	//sort first index array and second index array respectively
	//wrap device pointers
	thrust::device_ptr<int> dp_idx1 = thrust::device_pointer_cast(d_idx1);
	thrust::device_ptr<int> dp_idx2 = thrust::device_pointer_cast(d_idx2);

	thrust::sort_by_key(dp_idx1, dp_idx1 + kBest, dp_idx2); //sort ascending
#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	//clean up the mess
	cudaFree(d_x);
	cudaFree(d_values);
	culaSparseDestroyPlan(plan);
	culaSparseDestroy(handle);
}

//handles only device pointer.
culaSparseStatus CMEstimatorGPUSparse::computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config,
															unsigned int dim, unsigned int nnz, double* A, int* rowPtr, int* colIdx, double* x, double* b)
{
	// information returned by the solver
	culaSparseResult result;

	// associate coo data with the plan
	culaSparseSetDcsrData(handle, plan, 0, dim, nnz, A, rowPtr, colIdx, x, b);

	// execute plan
	culaSparseStatus status = culaSparseExecutePlan(handle, plan, &config, &result);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	//print if error
	if (culaSparseNoError != status)
	{
		char buffer[512];
		culaSparseGetResultString(handle, &result, buffer, 512);
		printf("%s\n", buffer);
	}



	return status;
}

void CMEstimatorGPUSparse::computeRandomComparisons(MatrixHandler* T, const int k)
{
	printf("0\t0\t");
	GPUSparse* matrix = dynamic_cast<GPUSparse*>(T);
	if (k != lastSize)
	{
		initIdxDevicePointers(k, matrix->getDimension());
		lastSize = k;
	}
	matrix->fillRandomCompareIndices(d_idx1, d_idx2, d_res, k);
}

