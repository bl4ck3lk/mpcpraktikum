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
 *      Author: Fabian
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

#define CUDA_CHECK_ERROR() {							\
    cudaError_t err = cudaGetLastError();					\
    if (cudaSuccess != err) {						\
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
                __FILE__, __LINE__, cudaGetErrorString(err) );	\
        exit(EXIT_FAILURE);						\
    }									\
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
static __global__ void initKernel(long* gpuIndices, double* x, const double* b, const int dim, const int columnIdx)
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

	res = NULL; //todo remove me (testing)
	idx1 = NULL; //todo remove me (testing)

	/* cula initialization */
	initCula();
}


CMEstimatorGPUSparse::~CMEstimatorGPUSparse() {
	//free device pointer
	if (d_idx1 != NULL) cudaFree(d_idx1);
	if (d_idx2 != NULL) cudaFree(d_idx2);
	if (d_res != NULL) cudaFree(d_res);
	if (res != NULL) free(res);
}

int* CMEstimatorGPUSparse::getIdx1DevicePtr()
{
	return d_idx1;
}

int* CMEstimatorGPUSparse::getIdx2DevicePtr()
{
	return d_idx2;
}

int* CMEstimatorGPUSparse::getResDevicePtr()
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

//todo remove me. only for testing purpose.
int* CMEstimatorGPUSparse::getResHostPtr(int dim)
{
	if (res != NULL) free(res);
	res = (int*)malloc(dim*sizeof(int));

	cudaMemcpy(res, d_res, dim*sizeof(int), cudaMemcpyDeviceToHost);
	return res;
}

//todo remove me. only for testing purpose.
int* CMEstimatorGPUSparse::getIdx1HostPtr(int dim)
{
	if (idx1 != NULL) free(idx1);
	idx1 = (int*)malloc(dim*sizeof(int));

	cudaMemcpy(idx1, d_idx1, dim*sizeof(int), cudaMemcpyDeviceToHost);
	return idx1;
}

//todo remove me. only for testing purpose
void CMEstimatorGPUSparse::setResDevicePtr(int* res, int dim)
{
	cudaMemcpy(d_res, res, dim*sizeof(int), cudaMemcpyHostToDevice);
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

	printf("[ESTIMATOR]: Device index arrays with size %i allocated.\n",size);
}

//Not needed in this particular implementation
Indices* CMEstimatorGPUSparse::getInitializationIndices(MatrixHandler* T, int initNr)
{
	return NULL;
}

//Not needed in this particular implementation.
Indices* CMEstimatorGPUSparse::getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBestForThisColumn)
{
	return NULL;
}


/*
 * Determines column-wise the best confidence measures of the specific column and saves it indices in two arrays.
 * Returns the number of actually written slots.
 */
int CMEstimatorGPUSparse::determineBestConfMeasures(double* xColumnDevice, double* bColumnDevice, int columnIdx, int dim, int kBest, int kBestForThisColumn, int currIndexNr)
{
	//debug //TODO remove me
	//printf("currentIndexNr = %i\n", currIndexNr);	


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

	//maybe recast pointers? (from thrust)
	//unsigned int * raw_ptr = thrust::raw_pointer_cast(dev_data_ptr);

	//save 'kBestForThisColumn' indices if possible (maybe not enough indices available)
	numBlocks = (kBest + THREADS - 1) / THREADS;
	int notWritten = 0;
	cudaMemcpyToSymbol(d_notWritten, &notWritten, sizeof(int));
	saveIndicesKernel<<<numBlocks, numThreads>>>(gpuIndices, d_idx1, d_idx2, dim, kBest, kBestForThisColumn, currIndexNr);
	cudaMemcpyFromSymbol(&notWritten, d_notWritten, sizeof(int));

//	printf("notWritten = %i\n", notWritten);

	CUDA_CHECK_ERROR();

	//free memory
	cudaFree(gpuIndices);

	return kBestForThisColumn - notWritten;
}

Indices* CMEstimatorGPUSparse::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	return NULL;
}

void CMEstimatorGPUSparse::getKBestConfMeasuresSparse(MatrixHandler* T, float* F, int kBest)
{
	printf("[ESTIMATOR]: Determine kBest confidence measures on GPU (column-wise).\n");

	//invoked only on sparse matrixhandler
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
	double* d_values = T_sparse->getValueArrDouble(true);
	int* d_colIdx = T_sparse->getColIdxDevice();
	int* d_rowPtr = T_sparse->getRowPtrDevice();

	//TODO remove debug stuff
//	double* d_valuesDHost = (double*)malloc(nnz*sizeof(double));
//	float* d_valsHost = (float*)malloc(nnz*sizeof(float));
//	cudaMemcpy(d_valsHost, d_values, nnz*sizeof(float), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < nnz; i++)
//	{
//		d_valuesDHost[i] = (double) d_valsHost[i];
//	}
//	double* d_valuesD;
//	cudaMalloc((void**) &d_valuesD, nnz*sizeof(double));
//	cudaMemcpy(d_valuesD, d_valuesDHost, nnz*sizeof(double), cudaMemcpyHostToDevice);
//	printf("double vals HOST\n");
//	Tester::printArrayDouble(d_valuesDHost, nnz);

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

	config.relativeTolerance = 1e-4;
	config.maxIterations = 50;
	config.maxRuntime = 1;
	config.useBestAnswer = 1;

	culaSparseCreate(&handle); //create library handle
	culaSparseConfigInit(handle, &config); //initialize values
	culaSparseCreatePlan(handle, &plan); //create execution plan
	culaSparseSetCudaDevicePlatform(handle, plan, 0); //use the CUDA-device platform (interprets given pointer as device pointers)
	culaSparseSetCgSolver(handle, plan, 0); //associate CG solver with the plan
	culaSparseSetJacobiPreconditioner(handle, plan, 0); //associate jacobi preconditioner with the plan

	int noError = 0;
	int solverTrials = 0;
	int determinedIndicesByNow = 0;
	//printf("[CMESTIMATOR]: Solve Eq. system column by column.\n");
	int start_s_CULA = clock();
	for(int i = 0; i < (dim-1) && countIndices < kBest; i++) //if enough values are gathered, stop computation
	{
		//0. determine number of best values for this column
		//The bigger i, the less best indices are determined for this column
		int xBestForThisColumn = ((dim-i)/(0.5*dim*(dim-1))) * kBest;
		if (!xBestForThisColumn) xBestForThisColumn = 1; //at least 1 per column
		//take into account that probably not as many indices as needed can be determined, so try o get them in the next column
		int determineXforThisColumn = xBestForThisColumn + (determinedIndicesByNow - countIndices);
		//1. Compute confidence measure for this column (solve Ax=b)

		d_b = T_sparse->getColumnDouble(i);

		//TODO remove debug copying stuff...
//		double* d_bD;
//		cudaMalloc((void**) &d_bD, dim*sizeof(double));
//		float* tmpHostFloat = (float*)malloc(dim*sizeof(float));
//		cudaMemcpy(tmpHostFloat, d_b, dim*sizeof(float), cudaMemcpyDeviceToHost);
//		double* tmpHost = (double*)malloc(dim*sizeof(double));
//		for(int k = 0; k < dim; k++)
//		{
//			tmpHost[k] = (double)tmpHostFloat[k];
//		}
//		cudaMemcpy(d_bD, tmpHost, dim*sizeof(double), cudaMemcpyHostToDevice);
//		free(tmpHost);
//		free(tmpHostFloat);

//		GPUSparse::printGpuArrayD(d_valuesD, nnz, "d vals double");
//		GPUSparse::printGpuArrayD(d_bD, dim, "d_bD");



		culaSparseStatus res = computeConfidenceMeasure(handle, plan, config, dim, nnz, d_values, d_rowPtr, d_colIdx, d_x, d_b);


		solverTrials++;
		if(res == culaSparseUnspecifiedError || res == culaSparseRuntimeError || res == culaSparseInteralError)
		{
			//A strange error occured, //TODO possibly handle

			if(res == culaSparseRuntimeError)
			{
				printf("cula Runtime Error\n");
			}

			printf("+++++++++++++++++++++++BREAK\n");
			T_sparse->print();
//			GPUSparse::printGpuArrayF(d_values, nnz, "A");
//			GPUSparse::printGpuArrayF(d_b, dim, "d_b");



			//TODO does happen only with cuda-memcheck??????????
			exit(1);
		}
		else
		{

			if(res == culaSparseNoError)
				noError++;


//			double* downloaded_x = GPUSparse::downloadGPUArrayDouble(d_x, dim);
//			double* downloaded_b = GPUSparse::downloadGPUArrayDouble(d_b, dim);
//			float* d_xF = (float*)malloc(dim*sizeof(float));
//			float* d_bF = (float*)malloc(dim*sizeof(float));
//			for(int k = 0; k < dim; k++)
//			{
//				d_xF[k] = (float)downloaded_x[k];
//				d_bF[k] = (float)downloaded_b[k];
//			}
//			float* reuploaded_x;
//			float* reuploaded_b;
//			cudaMalloc((void**) &reuploaded_x, sizeof(float)*dim);
//			cudaMemcpy(reuploaded_x, d_xF, sizeof(float)*dim, cudaMemcpyHostToDevice);
//			cudaMalloc((void**) &reuploaded_b, sizeof(float)*dim);
//			cudaMemcpy(reuploaded_b, d_bF, sizeof(float)*dim, cudaMemcpyHostToDevice);
//			Tester::testColumnSolution(GPUSparse::downloadGPUArrayInt(d_rowPtr, dim+1), GPUSparse::downloadGPUArrayInt(d_colIdx, nnz),
//								GPUSparse::downloadGPUArrayFloat(d_values, nnz), GPUSparse::downloadGPUArrayFloat(d_b, dim),
//								d_xF, i, dim);

//			printf("before dBCM countIndice = %i, xForThisCol = %i, determineXforThis = %i\n", countIndices,
//					xBestForThisColumn, determineXforThisColumn);
			//2. get indices of x best confidence measure values
			//printf("before determineBestConfMeasures\n");
//			int start_s=clock();
				// the code you wish to time goes here
			int writtenIndices = determineBestConfMeasures(d_x, d_b, i, dim, kBest, determineXforThisColumn, countIndices);
			countIndices += writtenIndices;

//			int stop_s=clock();
//					std::cout << "time dBCM: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << std::endl;

			//printf("\tafter dBCM; written = %i\n", writtenIndices);

		}

		cudaFree(d_b);

		determinedIndicesByNow += xBestForThisColumn; // #indices that should have been determined

//		printf("Column %i, try to determine %i best values. Actually determined by now %i values\n", i, determineXforThisColumn, countIndices);
	}
	int stop_s_CULA=clock();
			std::cout << "time Solving: " << (stop_s_CULA-start_s_CULA)/double(CLOCKS_PER_SEC)*1000 << std::endl;

	printf("After solving [%i of %i NO ERROR]! Going to sort with thrust\n", noError, solverTrials);
		//sort first index array and second index array respectively
		//wrap device pointers
		thrust::device_ptr<int> dp_idx1 = thrust::device_pointer_cast(d_idx1);
		thrust::device_ptr<int> dp_idx2 = thrust::device_pointer_cast(d_idx2);

		thrust::sort_by_key(dp_idx1, dp_idx1 + kBest, dp_idx2); //ascending
		CUDA_CHECK_ERROR();


	if (false) //debug printing
	{
		int* h_idx1 = new int[kBest];
		int* h_idx2 = new int[kBest];
		int* h_res = new int[kBest];

		cudaMemcpy(h_idx1, d_idx1, kBest*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_idx2, d_idx2, kBest*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_res, d_res, kBest*sizeof(int), cudaMemcpyDeviceToHost);

		printf("Images to be compared:\n");
		Tester::printArrayInt(h_idx1, kBest);
		Tester::printArrayInt(h_idx2, kBest);
		Tester::printArrayInt(h_res, kBest);
	}

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

//	printf("bla\n");

	// execute plan
	culaSparseStatus status = culaSparseExecutePlan(handle, plan, &config, &result);

	CUDA_CHECK_ERROR();

//	printf("blub\n");

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

