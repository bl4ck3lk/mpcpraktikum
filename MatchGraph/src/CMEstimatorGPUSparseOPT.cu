/*
 * CMEstimatorGPUSparseOPT.cu
 *
 * GPU implementation of finding the k-best image-pairs.
 * This implementation stores the k-best confidence measure values of a
 * certain amount of randomly chosen columns (depending on dimension and
 * number of requested image pairs) in the index arrays, managed by this
 * class.
 * These indices are generated column-wise with CULA solving the linear
 * equation system. For this, already stored memory from the matrix handler
 * is used.
 * The overall memory usage of this class is constant (4*k), depending on
 * the number of requested image-pairs k.
 * The resulting arrays contain only image-pairs that have not yet been
 * compared and they do not contain symmetric entries.
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

/*
 * Initialize index arrays with dim+1 and array for image-comparison result with 0.
 */
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

/*
 * Initialize buffer arrays.
 */
static __global__ void initTMPIndexArrays(long* d_tmpIndices, double* d_tmpConf, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
	{
		d_tmpIndices[idx] = -1;
		d_tmpConf[idx] = -DBL_MAX;
	}
}


/*
 * Initialize index array such that indices = [-1,1,2,-1,...,dim-1], whereas the respective
 * diagonal element is -1 as well as elements that are already compared or within the upper
 * diagonal matrix.
 * For already known elements (i.e. bColumnDevice[i] != 0), xColumnDevice[i] will be
 * assigned a very low value to prevent them from getting chosen later.
 */
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

/*
 * Store the first k-best values in buffer arrays.
 */
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

/*
 * Store the continuous indices of the buffer array in the dedicated output arrays.
 */
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

/*
 * Constructor
 */
CMEstimatorGPUSparseOPT::CMEstimatorGPUSparseOPT() {
	lastSize = 0;
	d_idx1 = NULL;
	d_idx2 = NULL;
	d_res = NULL;
	
	totalTime = 0; //runtime measurement

	//only for the debugging and testing functions (otherwise unused)
	res = NULL;
	idx1 = NULL;

	printf("Using estimator: CMEstimatorGPUSparseOPT\n");
}

/*
 * Destructor
 */
CMEstimatorGPUSparseOPT::~CMEstimatorGPUSparseOPT() {
	//free device pointer
	if (d_idx1 != NULL) cudaFree(d_idx1);
	if (d_idx2 != NULL) cudaFree(d_idx2);
	if (d_res != NULL) cudaFree(d_res);
	if (res != NULL) free(res);

	printf("Total solver time:%f\n", totalTime*(1/(double)1000000000));
}

/*
 * Return pointer to device index array1 (i-th index).
 */
int* CMEstimatorGPUSparseOPT::getIdx1Ptr()
{
	return d_idx1;
}

/*
 * Return pointer to device index array2 (j-th index).
 */
int* CMEstimatorGPUSparseOPT::getIdx2Ptr()
{
	return d_idx2;
}

/*
 * Return pointer to device array for image-comparison result.
 */
int* CMEstimatorGPUSparseOPT::getResPtr()
{
	return d_res;
}

/*
 * Function for debugging and testing.
 * Returns pointer to host array containing the device memory for the
 * image-comparison results.
 */
int* CMEstimatorGPUSparseOPT::getResHostPtr(int dim)
{
	if (res != NULL) free(res);
	res = (int*)malloc(dim*sizeof(int));

	cudaMemcpy(res, d_res, dim*sizeof(int), cudaMemcpyDeviceToHost);
	return res;
}

/*
 * Function for debugging and testing.
 * Returns pointer to host array containing the device memory for the
 * index array1.
 */
int* CMEstimatorGPUSparseOPT::getIdx1HostPtr(int dim)
{
	if (idx1 != NULL) free(idx1);
	idx1 = (int*)malloc(dim*sizeof(int));

	cudaMemcpy(idx1, d_idx1, dim*sizeof(int), cudaMemcpyDeviceToHost);
	return idx1;
}

/*
 * Function for debugging and testing.
 * Function to set the device memory for the image-comparison result
 * to a specific content.
 */
void CMEstimatorGPUSparseOPT::setResDevicePtr(int* res, int dim)
{
	cudaMemcpy(d_res, res, dim*sizeof(int), cudaMemcpyHostToDevice);
}

/*
 * Allocate resulting arrays.
 */
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

	//printf("[ESTIMATOR]: Device index arrays with size %i allocated.\n",size);
}

/*
 * Determines column-wise the best confidence measures of the specific column and saves k-best image pairs
 * in a buffer.
 */
void CMEstimatorGPUSparseOPT::determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, long* d_tmpIndices, double* d_tmpConfidences, int dim, int kBest)
{
	//Allocate index array on GPU
	long* gpuIndices;
	cudaMalloc((void**) &gpuIndices, dim * sizeof(long));

	//wrap raw pointer with device pointer
	thrust::device_ptr<long> dp_gpuIndices = thrust::device_pointer_cast(gpuIndices);

	//kernel settings for index array
	int numBlocks = (dim + THREADS - 1) / THREADS;
	int numThreads = THREADS;

	//initializes index array and data to meet the requirements (see class description for details)
	initKernel<<<numBlocks, numThreads>>>(gpuIndices, xColumnDevice, bColumnDevice, dim, columnIdx);
#if CHECK_FOR_CUDA_ERROR
  CUDA_CHECK_ERROR() 
#endif

	//wrap column device pointer
	thrust::device_ptr<double> dp_xColumn = thrust::device_pointer_cast(xColumnDevice);

	//sort x column and index array respectively
	//already known values will be the last ones due to initialization
	thrust::sort_by_key(dp_xColumn, dp_xColumn + dim, dp_gpuIndices, thrust::greater<double>());

	//save the first kBest values in the tmp arrays, after the best values by now
	numBlocks = (kBest + THREADS - 1) / THREADS;
	saveTMPindicesKernel<<<numBlocks, numThreads>>>(gpuIndices, d_tmpIndices, d_tmpConfidences, xColumnDevice, dim, kBest);
#if CHECK_FOR_CUDA_ERROR
  CUDA_CHECK_ERROR() 
#endif

	//free memory
	cudaFree(gpuIndices);
}

/*
 * Find the k-best image-pairs in the given confidence measure matrix (of some randomly chosen
 * columns). These image-pairs are stored in the allocated index arrays on device.
 * The confidence measure matrix argument F is not used (i.e. NULL) in the case of using
 * the GPU implementation. The confidence measures are solved implicit with CULA while
 * calling this function.
 */
void CMEstimatorGPUSparseOPT::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	//printf("[ESTIMATOR]: Determine kBest confidence measures on GPU (randomly picking columns).\n");

	//invoked only on sparse MatrixHandler
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);
	unsigned int dim = T_sparse->getDimension();

	//buffer device memory to gather the overall maxima
	long* d_tmpIndices;
	double* d_tmpConfidences;
	cudaMalloc((void**) &d_tmpIndices, 2*kBest*sizeof(long));
	cudaMalloc((void**) &d_tmpConfidences, 2*kBest*sizeof(double));
	int numBlocks = ((2*kBest) + THREADS - 1) / THREADS;
	initTMPIndexArrays<<<numBlocks, THREADS>>>(d_tmpIndices, d_tmpConfidences, 2*kBest);
#if CHECK_FOR_CUDA_ERROR
  CUDA_CHECK_ERROR() 
#endif

	/*
	 * if index array size changed since last use, allocate new device memory
	 * with new size and free old device memory. Otherwise reuse device memory.
	 */
	if (kBest != lastSize)
	{
		initIdxDevicePointers(kBest, dim);
		lastSize = kBest;
	}

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
	config.relativeTolerance = 1e-6;
	config.maxIterations = 500;
	//config.maxRuntime = 1;
	//config.useBestAnswer = 1;
	culaSparseSetCgSolver(handle, plan, 0); //associate CG solver with the plan
	culaSparseSetJacobiPreconditioner(handle, plan, 0); //associate jacobi preconditioner with the plan

	//calculate number of randomly chosen columns and number of elements chosen per column
	double ddim = double(dim);
	double f = (double(kBest)/ddim) < 0.5 ? 0.5 : (double(kBest)/ddim);
	
	//printf("Dim: %i \t kBest: %i\n",dim,kBest);
	double perCol =  double(kBest)/(ddim*pow(ddim, 0.55))*(exp(-(1/(f*ddim))*ddim)*ddim);
	int xBestForThisColumn = ceil(perCol);
	if (!xBestForThisColumn) xBestForThisColumn = 1; //at least 1 per column
	int nrCols = (kBest/xBestForThisColumn) + 2; //round-off error
	//printf("Per column:%i    Nr Cols: %i\n",xBestForThisColumn, nrCols);

	//buffer for already searched columns
	char* colsVisited = (char*)malloc(sizeof(char)*dim);
	memset(colsVisited, 0, dim);

	//first random column
	int column = rand() % dim;

	//counter for the number of image-pairs that have been chosen
	int countIndices = 0;

	//number of image-pairs that should have been chosen by now
	int determinedIndicesByNow = 0;

	int noError = 0; //counter for solved columns with CULA error
	int solverTrials = 0; //number of solved columns.

	__int64_t startCula = continuousTimeNs(); //start CULA measurement
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
				printf("CULA runtime error\n");
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
#if CHECK_FOR_CUDA_ERROR
  CUDA_CHECK_ERROR() 
#endif
		}

		determinedIndicesByNow += xBestForThisColumn; //number of indices that should have been determined

		//choose another random column that has not yet been chosen
		column = rand() % dim;
		const int startCol = column;
		while(colsVisited[column] == 1)
		{
			column = (column + 1) % dim;
			if(column == startCol)
				break;
		}

		//printf("Column %i, try to determine %i best values. Actually determined by now %i values\n", i, determineXforThisColumn, countIndices);
	}
	__int64_t solverDiff = continuousTimeNs()-startCula; //stop CULA measurement
	totalTime += solverDiff; //CULA runtime
	printf("%f\t%i\t", solverDiff*(1/(double)1000000000), (solverTrials-noError));

	//store results from buffer in the actual resulting arrays
	numBlocks = (kBest + THREADS - 1) / THREADS;
	mergeInOutIndiceArrays<<<numBlocks, THREADS>>>(d_tmpIndices, d_idx1, d_idx2, kBest, dim);

	//sort first index array and second index array respectively
	//wrap device pointers
	thrust::device_ptr<int> dp_idx1 = thrust::device_pointer_cast(d_idx1);
	thrust::device_ptr<int> dp_idx2 = thrust::device_pointer_cast(d_idx2);

	thrust::sort_by_key(dp_idx1, dp_idx1 + kBest, dp_idx2); //sort ascending
#if CHECK_FOR_CUDA_ERROR
  CUDA_CHECK_ERROR() 
#endif

	//clean up the mess.
	free(colsVisited);
	cudaFree(d_x);
	cudaFree(d_values);
	cudaFree(d_tmpIndices);
	cudaFree(d_tmpConfidences);
	culaSparseDestroyPlan(plan);
	culaSparseDestroy(handle);
}

/*
 * Solve the linear equation system of a specific column with CULA.
 * Due to the prior set-up, only device pointers can be used here.
 */
culaSparseStatus CMEstimatorGPUSparseOPT::computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config,
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
		//printf("%s\n", buffer);
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

/*
 * Fill index array1 and index array2 with random image pairs for random
 * iteration.
 */
void CMEstimatorGPUSparseOPT::computeRandomComparisons(MatrixHandler* T, const int k)
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

