/*
 * CMEstimatorGPUSparseOPT.cu
 *
 * Generates a list of indices containing the i, j index of the overall
 * k-best confidence measure values in the lower diagonal matrix. (Slow)
 * The List of indices is generated column wise after Cula solved the
 * linear equation system. This class uses the already stored memory
 * of the equation solver and extracts the k best values of the specific
 * column.
 *
 *  Created on: 16.07.2013
 *      Author: Fabian, Armin
 */

#include "GPUSparse.h"
#include "CMEstimatorGPUSparseOPT.h"
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
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>       /* exp */

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

//Initialize TMP index arrays
static __global__ void initTMPIndexArrays(long* d_tmpIndices, double* d_tmpConf, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		d_tmpIndices[idx] = -1;
		d_tmpConf[idx] = -DBL_MAX;
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

//Save the first kBest-Values in tmp Arrays
static __global__ void saveTMPindicesKernel(long* gpuIndices, long* d_tmpInd, double* d_tmpConf, double* d_x, int dim, int kBest)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int write_idx = t_idx + kBest;

	if (t_idx < dim && write_idx < 2*kBest)
	{
		int gpuIdx = gpuIndices[t_idx]; //size(gpuIndices) > kBestForThisColumn > t_idx
		double conf = d_x[t_idx];

		d_tmpInd[write_idx] = gpuIdx;
		d_tmpConf[write_idx] = conf;
	}
}

//Save the continuous indices in the dedicated output arrays
static __global__ void mergeInOutIndiceArrays(long* d_tmpInd, int* d_idx1, int* d_idx2, int kBest, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < kBest)
	{
		long tmpInd = d_tmpInd[idx];
		d_idx1[idx] = tmpInd/dim;
		d_idx2[idx] = tmpInd%dim;
	}
}

CMEstimatorGPUSparseOPT::CMEstimatorGPUSparseOPT() {
	lastSize = 0;
	d_idx1 = NULL;
	d_idx2 = NULL;
	d_res = NULL;

	res = NULL; //todo remove me (testing)
	idx1 = NULL; //todo remove me (testing)
	printf("Using estimator: CMEstimatorGPUSparseOPT\n");
}


CMEstimatorGPUSparseOPT::~CMEstimatorGPUSparseOPT() {
	//free device pointer
	if (d_idx1 != NULL) cudaFree(d_idx1);
	if (d_idx2 != NULL) cudaFree(d_idx2);
	if (d_res != NULL) cudaFree(d_res);
	if (res != NULL) free(res);
}

int* CMEstimatorGPUSparseOPT::getIdx1Ptr()
{
	return d_idx1;
}

int* CMEstimatorGPUSparseOPT::getIdx2Ptr()
{
	return d_idx2;
}

int* CMEstimatorGPUSparseOPT::getResPtr()
{
	return d_res;
}

//todo remove me. only for testing purpose.
int* CMEstimatorGPUSparseOPT::getResHostPtr(int dim)
{
	if (res != NULL) free(res);
	res = (int*)malloc(dim*sizeof(int));

	cudaMemcpy(res, d_res, dim*sizeof(int), cudaMemcpyDeviceToHost);
	return res;
}

//todo remove me. only for testing purpose.
int* CMEstimatorGPUSparseOPT::getIdx1HostPtr(int dim)
{
	if (idx1 != NULL) free(idx1);
	idx1 = (int*)malloc(dim*sizeof(int));

	cudaMemcpy(idx1, d_idx1, dim*sizeof(int), cudaMemcpyDeviceToHost);
	return idx1;
}

//todo remove me. only for testing purpose
void CMEstimatorGPUSparseOPT::setResDevicePtr(int* res, int dim)
{
	cudaMemcpy(d_res, res, dim*sizeof(int), cudaMemcpyHostToDevice);
}

//Allocate device memory for index pointers and clear last used pointers
//(for dynamic change of kBest values index-arays)
void CMEstimatorGPUSparseOPT::initIdxDevicePointers(int size, unsigned int dim)
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

	printf("[ESTIMATOR]: Device index arrays with size %i allocated.\n",size);
}

/*
 * Determines column-wise the best confidence measures of the specific column and saves it indices in two arrays.
 * Returns the number of actually written slots.
 */
void CMEstimatorGPUSparseOPT::determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, long* d_tmpIndices, double* d_tmpConfidences, int dim, int kBest)
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
	CUDA_CHECK_ERROR();

	//wrap column device pointer
	thrust::device_ptr<double> dp_xColumn = thrust::device_pointer_cast(xColumnDevice);

	//sort x column and index array respectively
	//already known values will be the last ones due to initialization
	thrust::sort_by_key(dp_xColumn, dp_xColumn + dim, dp_gpuIndices, thrust::greater<double>());
	CUDA_CHECK_ERROR();

	//save the first kBest values in the tmp arrays, after the best values by now
	numBlocks = (kBest + THREADS - 1) / THREADS;
	saveTMPindicesKernel<<<numBlocks, numThreads>>>(gpuIndices, d_tmpIndices, d_tmpConfidences, xColumnDevice, dim, kBest);
	CUDA_CHECK_ERROR();

	//free memory
	cudaFree(gpuIndices);
}

void CMEstimatorGPUSparseOPT::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("[ESTIMATOR]: Determine kBest confidence measures on GPU (randomly picking columns).\n");

	//invoked only on sparse MatrixHandler
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);
	unsigned int dim = T_sparse->getDimension();

	//tmp device memoy to gather the overall maxima
	long* d_tmpIndices;
	double* d_tmpConfidences;
	cudaMalloc((void**) &d_tmpIndices, 2*kBest*sizeof(long));
	cudaMalloc((void**) &d_tmpConfidences, 2*kBest*sizeof(double));
	int numBlocks = ((2*kBest) + THREADS - 1) / THREADS;
	initTMPIndexArrays<<<numBlocks, THREADS>>>(d_tmpIndices, d_tmpConfidences, 2*kBest);
	CUDA_CHECK_ERROR()

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

	//Calcuate n of columns and elements per column
	double f = double(kBest)/double(dim);
//	printf("Dim: %i \t kBest: %i\n",dim,kBest);
	double perCol =  double(kBest)/(exp(-(1/(f*double(dim)))*double(dim))*double(dim));
	int xBestForThisColumn = ceil(perCol);
	if (!xBestForThisColumn) xBestForThisColumn = 1; //at least 1 per column
	int nrCols = (kBest/xBestForThisColumn) + 2; //round-off error
//	printf("Per column:%i    Nr Cols: %i\n",xBestForThisColumn, nrCols);
	char* colsVisited = (char*)malloc(sizeof(char)*dim);
	memset(colsVisited, 0, dim);

	int column = rand() % dim;
	int determinedIndicesByNow = 0;

	int noError = 0;
	int solverTrials = 0;
	//printf("[CMESTIMATOR]: Solve Eq. system column by column.\n");
//	int start_s_CULA = clock();
	for(int i = 0; i < nrCols && countIndices < kBest; i++) //if enough values are gathered, stop computation
	{
		//1. Compute confidence measure for this column (solve Ax=b)
		d_b = T_sparse->getColumnDouble(column);
		culaSparseStatus res = computeConfidenceMeasure(handle, plan, config, dim, nnz, d_values, d_rowPtr, d_colIdx, d_x, d_b);
		colsVisited[column] = 1;

		solverTrials++;
		if(res == culaSparseUnspecifiedError || res == culaSparseRuntimeError || res == culaSparseInteralError)
		{
			//A bad CULA error occurred

			if(res == culaSparseRuntimeError)
			{
				printf("CULA Runtime Error\n");
			}

			printf("Exiting due to CULA ERROR!\n");
			T_sparse->print();

			exit(EXIT_FAILURE);
		}
		else
		{
			if(res == culaSparseNoError)
				noError++;

			//2. get indices of x best confidence measure values
			determineBestConfMeasures(d_x, d_b, column, d_tmpIndices, d_tmpConfidences, dim, kBest);

			//sort TMP index array such that the highest confidence measures are at the front
			thrust::device_ptr<long> dp_tmpIndices = thrust::device_pointer_cast(d_tmpIndices);
			thrust::device_ptr<double> dp_tmpConfidences = thrust::device_pointer_cast(d_tmpConfidences);
			thrust::sort_by_key(dp_tmpConfidences, dp_tmpConfidences + 2*kBest, dp_tmpIndices, thrust::greater<double>());
			CUDA_CHECK_ERROR();
		}

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

//	int stop_s_CULA=clock();
//	std::cout << "time Solving: " << (stop_s_CULA-start_s_CULA)/double(CLOCKS_PER_SEC)*1000 << std::endl;
//	printf("After solving [%i of %i NO ERROR]! Going to sort with thrust\n", noError, solverTrials);

	numBlocks = (kBest + THREADS - 1) / THREADS;
	mergeInOutIndiceArrays<<<numBlocks, THREADS>>>(d_tmpIndices, d_idx1, d_idx2, kBest, dim);

	//sort first index array and second index array respectively
	//wrap device pointers
	thrust::device_ptr<int> dp_idx1 = thrust::device_pointer_cast(d_idx1);
	thrust::device_ptr<int> dp_idx2 = thrust::device_pointer_cast(d_idx2);

	thrust::sort_by_key(dp_idx1, dp_idx1 + kBest, dp_idx2); //sort ascending
	CUDA_CHECK_ERROR()

	if (false) //debug printing
	{
		Helper::printGpuArray(d_idx1, kBest, "Idx1");
		Helper::printGpuArray(d_idx2, kBest, "Idx2");
	}

	//clean up the mess.
	free(colsVisited);
	cudaFree(d_x);
	cudaFree(d_values);
	cudaFree(d_tmpIndices);
	cudaFree(d_tmpConfidences);
	culaSparseDestroyPlan(plan);
	culaSparseDestroy(handle);
}

//handles only device pointer.
culaSparseStatus CMEstimatorGPUSparseOPT::computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config,
															unsigned int dim, unsigned int nnz, double* A, int* rowPtr, int* colIdx, double* x, double* b)
{
	// information returned by the solver
	culaSparseResult result;

	// associate coo data with the plan
	culaSparseSetDcsrData(handle, plan, 0, dim, nnz, A, rowPtr, colIdx, x, b);

	// execute plan
	culaSparseStatus status = culaSparseExecutePlan(handle, plan, &config, &result);
	CUDA_CHECK_ERROR();

	//print if error
	if (culaSparseNoError != status)
	{
		char buffer[512];
		culaSparseGetResultString(handle, &result, buffer, 512);
		printf("%s\n", buffer);
	}

	//print resulting vector x if needed
	if (false)
	{
		float* h_x = new float[dim];
		cudaMemcpy(h_x, x, dim * sizeof(float), cudaMemcpyDeviceToHost);
		printf("X = [");
		for (int i = 0; i < dim; i++)
		{
			printf(" %f ", h_x[i]);
		}
		printf("]\n");
	}
	return status;
}

void CMEstimatorGPUSparseOPT::computeRandomComparisons(MatrixHandler* T, const int k)
{
	//todo
}

