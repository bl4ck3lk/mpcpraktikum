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

const int THREADS = 64;

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
static __global__ void initKernel(long* gpuIndices, float* x, const float* b, const int dim, const int columnIdx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < dim)
	{
		if (columnIdx >= idx || 0 != b[idx]) //diagonal element or known element or upper diagonal matrix element
		{
			gpuIndices[idx] = -1;
			//assign very low value to avoid them getting chosen
			x[idx] = -FLT_MAX;
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

	if (t_idx < dim && write_idx < kBest)
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
}


CMEstimatorGPUSparse::~CMEstimatorGPUSparse() {
	//free device pointer
	if (d_idx1 != NULL) cudaFree(d_idx1);
	if (d_idx2 != NULL) cudaFree(d_idx2);
	if (d_res != NULL) cudaFree(d_res);
}


//Allocate device memory for index pointers and clear last used pointers
//(for dynamic change of kBes values index-arays)
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
	dim3 threadBlock(THREADS);
	dim3 blockGrid(numBlocks);
	initIndexArrays<<<blockGrid, threadBlock>>>(d_idx1, d_idx2, d_res, size, dim);

	//todo remove debug printing
	int* testResult1 = new int[size];
	int* testResult2 = new int[size];
	int* testResult3 = new int[size];
	cudaMemcpy(testResult1, d_idx1, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult2, d_idx2, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(testResult3, d_res, size*sizeof(int), cudaMemcpyDeviceToHost);
	Tester::printArrayInt(testResult1, size);
	Tester::printArrayInt(testResult2, size);
	Tester::printArrayInt(testResult3, size);

	printf("[ESTIMATOR]: Device index arrays with size %i allocated.\n",size);
}

//todo remove me
Indices* CMEstimatorGPUSparse::getInitializationIndices(MatrixHandler* T, int initNr)
{
	Indices* initIndices = new Indices[initNr];
	std::vector<int> chosenOnes; //max size will be initNr
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
		} while ( ((rIdx < 1+(rIdx/dim)+(rIdx/dim)*dim)
					|| (T->getVal(x,y) != 0)
					|| (std::find(chosenOnes.begin(), chosenOnes.end(), rIdx) != chosenOnes.end()))
				&& (c <= MAX_ITERATIONS) );
		/* As long as the random number is not within the upper diagonal matrix w/o diagonal elements
		 * or T(idx) != 0 generate or already in the list of Indices, a new random index but maximal
		 * MAX_ITERAtION times.
		 */

		if (c <= MAX_ITERATIONS) //otherwise initIndices contains -1 per struct definition
		{
			chosenOnes.push_back(rIdx);
			initIndices[i].i = x;
			initIndices[i].j = y;
		}
	}

	return initIndices;
}

//A*x_i = b_i
//todo remove me!
Indices* CMEstimatorGPUSparse::getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBestForThisColumn)
{
	//storage for the kBest indices
	//TODO remove me
	Indices* kBestIndices = new Indices[kBestForThisColumn];

	//Allocate index array on GPU
	long* gpuIndices;
	cudaMalloc((void**) &gpuIndices, dim * sizeof(long));
	CUDA_CHECK_ERROR();
	//wrap raw pointer with device pointer
	thrust::device_ptr<long> dp_gpuIndices = thrust::device_pointer_cast(gpuIndices);
	CUDA_CHECK_ERROR();

	//Kernel settings for index array
	int numBlocks = (dim + THREADS - 1) / THREADS;
	dim3 threadBlock(THREADS);
	dim3 blockGrid(numBlocks);

	/* Init indices array such that indices = [-1,1,2,-1,...,dim-1], whereas the respective
	 * diagonal element is -1 as well as elements that are already compared or within the upper
	 * diagonal matrix.
	 * For already known elements (i.e. bColumnDevice[i] != 0), xColumnDevice[i] will be
	 * assigned a very low value to prevent them from getting chosen later.
	 */
	initKernel<<<blockGrid, threadBlock>>>(gpuIndices, xColumnDevice, bColumnDevice, dim, columnIdx);
	CUDA_CHECK_ERROR();

	//wrap column device pointer
	thrust::device_ptr<float> dp_xColumn = thrust::device_pointer_cast(xColumnDevice);
	CUDA_CHECK_ERROR();

	//sort x column and index array respectively
	//already known values will be the last ones due to initialization
	thrust::sort_by_key(dp_xColumn, dp_xColumn + dim, dp_gpuIndices, thrust::greater<float>());
	CUDA_CHECK_ERROR();

	//download device memory
	long* indices = new long[kBestForThisColumn]; //at most kBest indices are needed
	//the first kBest indices are also the best conf. measure values after sorting
	thrust::copy(dp_gpuIndices, dp_gpuIndices + kBestForThisColumn, indices);
	CUDA_CHECK_ERROR();

	//free memory
	cudaFree(gpuIndices);

	//build indices list structure
	for(int i = 0; i<kBestForThisColumn; i++)
	{
		long idx = indices[i];
		if (indices[i] > -1)
		{
			kBestIndices[i].i = idx/dim;
			kBestIndices[i].j = idx%dim;
		}
		else
		{
			//after the first index with -1 all following
			//will contain -1.
			break;
		}
		//if some of the indices contained -1, the remaining
		//kBestIndices will contain also -1 as i,j index per
		//struct definition.
	}

	return kBestIndices;
}


/*
 * Determines column-wise the best confidence measures of the specific column and saves it indices in two arrays.
 * Returns the number of actually written slots.
 */
int CMEstimatorGPUSparse::determineBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest, int kBestForThisColumn, int currIndexNr)
{
	//Allocate index array on GPU
	long* gpuIndices;
	cudaMalloc((void**) &gpuIndices, dim * sizeof(long));
	CUDA_CHECK_ERROR();
	//wrap raw pointer with device pointer
	thrust::device_ptr<long> dp_gpuIndices = thrust::device_pointer_cast(gpuIndices);
	CUDA_CHECK_ERROR();

	//Kernel settings for index array
	int numBlocks = (dim + THREADS - 1) / THREADS;
	dim3 threadBlock(THREADS);
	dim3 blockGrid(numBlocks);

	/* Init indices array such that indices = [-1,1,2,-1,...,dim-1], whereas the respective
	 * diagonal element is -1 as well as elements that are already compared or within the upper
	 * diagonal matrix.
	 * For already known elements (i.e. bColumnDevice[i] != 0), xColumnDevice[i] will be
	 * assigned a very low value to prevent them from getting chosen later.
	 */
	initKernel<<<blockGrid, threadBlock>>>(gpuIndices, xColumnDevice, bColumnDevice, dim, columnIdx);
	CUDA_CHECK_ERROR();

	//wrap column device pointer
	thrust::device_ptr<float> dp_xColumn = thrust::device_pointer_cast(xColumnDevice);
	CUDA_CHECK_ERROR();

	//sort x column and index array respectively
	//already known values will be the last ones due to initialization
	thrust::sort_by_key(dp_xColumn, dp_xColumn + dim, dp_gpuIndices, thrust::greater<float>());
	CUDA_CHECK_ERROR();

	//maybe recast pointers? (from thrust)
	//unsigned int * raw_ptr = thrust::raw_pointer_cast(dev_data_ptr);

	//save 'kBestForThisColumn' indices if possible (maybe not enough indices available)
	numBlocks = (kBest + THREADS - 1) / THREADS;
	dim3 blockGrid2(numBlocks);
	int notWritten = 0;
	cudaMemcpyToSymbol(d_notWritten, &notWritten, sizeof(int));
	saveIndicesKernel<<<blockGrid2, threadBlock>>>(gpuIndices, d_idx1, d_idx2, dim, kBest, kBestForThisColumn, currIndexNr);
	cudaMemcpyFromSymbol(&notWritten, d_notWritten, sizeof(int));

	//free memory
	cudaFree(gpuIndices);

	return kBestForThisColumn - notWritten;
}



Indices* CMEstimatorGPUSparse::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("Determine kBest confidence measures on GPU (column-wise):\n");
	bool newMethod = true;// todo remove me

	//invoked only on sparse matrixhandler
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);
	unsigned int dim = T_sparse->getDimension();

	//indices cache
	//TODO not needed remove me
	Indices* bestIndices = new Indices[kBest];

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

	float* d_values = T_sparse->getValueArr(true);

	int* d_colIdx = T_sparse->getColIdxDevice();
	int* d_rowPtr = T_sparse->getRowPtrDevice();

	//GPUSparse::printGpuArrayF(d_values, nnz, "[ESTIMATOR] Values"); CUDA_CHECK_ERROR()
	//GPUSparse::printGpuArray(d_rowPtr, dim+1, "[ESTIMATOR] RowPtr"); CUDA_CHECK_ERROR()
	//GPUSparse::printGpuArray(d_colIdx, nnz, "[ESTIMATOR] colidx"); CUDA_CHECK_ERROR()

	//x-vector
	float* d_x;
	cudaMalloc((void**) &d_x, dim * sizeof(float));

	//b-vector
	float* d_b;
//	cudaMalloc((void**) &d_b, dim * sizeof(float));

	//*****************************************************
	// TODO directly obtain device pointers from GPUSparseB
//	d_values = T_sparse->getValueArr(true);

//	int* colIdx = T_sparse->getColIdx();
//	printf("[CMESTIMATOR]: ColIdx\n");
//	Tester::printArrayInt(colIdx, nnz);
//	cudaMalloc((void**) &d_colIdx, nnz * sizeof(int));
//	cudaMemcpy(d_colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);

//	int* rowPtr = T_sparse->getRowPtr();
//	printf("[CMESTIMATOR]: rowPtr\n");
//	Tester::printArrayInt(rowPtr, dim+1);
//	cudaMalloc((void**) &d_rowPtr, (dim+1) * sizeof(int));
//	cudaMemcpy(d_rowPtr, rowPtr, (dim+1) * sizeof(int), cudaMemcpyHostToDevice);
	// END *************************************************

//	printf("pointers\n");

	//Set up cula
	//TODO Try to move to constructor

	culaSparseHandle handle;
	culaSparsePlan plan;
	culaSparseConfig config;

	config.relativeTolerance = 1e-6;
	config.maxIterations = 300;
	config.maxRuntime = 10;

	culaSparseCreate(&handle); //create library handle
	culaSparseConfigInit(handle, &config); //initialize values
	culaSparseCreatePlan(handle, &plan); //create execution plan
	culaSparseSetCudaDevicePlatform(handle, plan, 0); //use the CUDA-device platform (interprets given pointer as device pointers)
	culaSparseSetCgSolver(handle, plan, 0); //associate CG solver with the plan
	culaSparseSetJacobiPreconditioner(handle, plan, 0); //associate jacobi preconditioner with the plan


	int determinedIndicesByNow = 0;
	printf("[CMESTIMATOR]: Solve Eq. system column by column.\n");
	for(int i = 0; i < dim && countIndices < kBest; i++) //if enough values are gathered, stop computation
	{
		//0. determine number of best values for this column
		//The bigger i, the less best indices are determined for this column
		int xBestForThisColumn = ((dim-i)/(0.5*dim*(dim-1))) * kBest;
		if (!xBestForThisColumn) xBestForThisColumn = 1; //at least 1 per column
		//take into account that probably not as many indices as needed can be determined, so try o get them in the next column
		int determineXforThisColumn = xBestForThisColumn + (determinedIndicesByNow - countIndices);
		Indices* tmpIndices = new Indices[determineXforThisColumn];
//		printf("[CMESTIMATOR]: Starting upload\n");
		//1. Compute confidence measure for this column (solve Ax=b)
//		printf("[CMESTIMATOR]: b_column\n");
		d_b = T_sparse->getColumn(i);
//		Tester::printArrayFloat(T_sparse->getColumn(i), dim);
//		cudaMemcpy(d_b, T_sparse->getColumn(i), dim * sizeof(float), cudaMemcpyHostToDevice);
//		printf("[CMESTIMATOR]: Uploaded column %i.\n",i);

		//debug
//		T_sparse->print();
//		GPUSparse::printGpuArrayF(d_values, nnz, "[ESTIMATOR] Values"); CUDA_CHECK_ERROR()
//		GPUSparse::printGpuArray(d_rowPtr, dim+1, "[ESTIMATOR] RowPtr"); CUDA_CHECK_ERROR()
//		GPUSparse::printGpuArray(d_colIdx, nnz, "[ESTIMATOR] colidx"); CUDA_CHECK_ERROR()
//		GPUSparse::printGpuArrayF(d_b, dim, "[ESTIMATOR] b"); CUDA_CHECK_ERROR()

		computeConfidenceMeasure(handle, plan, config, dim, nnz, d_values, d_rowPtr, d_colIdx, d_x, d_b);
//		printf("[CMESTIMATOR]: Solved column %i.\n",i);
		CUDA_CHECK_ERROR()

		//2. get indices of x best confidence measure values
		if (newMethod)
		{
			int writtenIndices = determineBestConfMeasures(d_x, d_b, i, dim, kBest, determineXforThisColumn, countIndices);
			countIndices += writtenIndices;
		}
		else
		{
			//todo remove me
			tmpIndices = getKBestConfMeasures(d_x, d_b, i, dim, determineXforThisColumn);
			//3. gather indices //todo remove me
			for(int j = 0; j < determineXforThisColumn && countIndices < kBest; j++)
			{
				if (-1 == tmpIndices[j].i) break; //following indices are also -1
				else
				{
					bestIndices[countIndices] = tmpIndices[j];
					countIndices++;
				}
			}
		}

		determinedIndicesByNow += xBestForThisColumn; // #indices that should have been determined

//		printf("Column %i, try to determine %i best values. Actually determined by now %i values\n", i, determineXforThisColumn, countIndices);
	}

	if (newMethod)
	{
		//sort first index array and second index array respectively
		//wrap device pointers
		thrust::device_ptr<int> dp_idx1 = thrust::device_pointer_cast(d_idx1);
		thrust::device_ptr<int> dp_idx2 = thrust::device_pointer_cast(d_idx2);
		CUDA_CHECK_ERROR();

		thrust::sort_by_key(dp_idx1, dp_idx1 + kBest, dp_idx2); //ascending
		CUDA_CHECK_ERROR();

		/*And of course, you can get your raw pointers back if you need to use them in a regular CUDA kernel afterward:
		unsigned int * raw_ptr = thrust::raw_pointer_cast(dev_data_ptr);
		*/
	}

	if (newMethod) //debug printing
	{
		int* h_idx1 = new int[kBest];
		int* h_idx2 = new int[kBest];
		int* h_res = new int[kBest];

		cudaMemcpy(h_idx1, d_idx1, kBest*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_idx2, d_idx2, kBest*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_res, d_res, kBest*sizeof(int), cudaMemcpyDeviceToHost);

		printf("Images to be compared:")
		Tester::printArrayInt(h_idx1, kBest);
		Tester::printArrayInt(h_idx2, kBest);
		Tester::printArrayInt(h_res, kBest);
	}

	//clean up the mess
	cudaFree(d_x);
	cudaFree(d_b);
	culaSparseDestroyPlan(plan);
	culaSparseDestroy(handle);

	//print
	if (!newMethod)
	{
		printf("%i best entries:\n", kBest);
		for(int i = 0; i < kBest; i++)
		{
			//value can't be printed because it is not saved in the Indices-list
			printf("%i: at [%i,%i]\n",i,bestIndices[i].i,bestIndices[i].j);
		}
	}

	return bestIndices;
}

//handles only device pointer.
void CMEstimatorGPUSparse::computeConfidenceMeasure(culaSparseHandle handle, culaSparsePlan plan, culaSparseConfig config,
															unsigned int dim, unsigned int nnz, float* A, int* rowPtr, int* colIdx, float* x, float* b)
{
	// information returned by the solver
	culaSparseResult result;

	// associate coo data with the plan
	culaSparseSetScsrData(handle, plan, 0, dim, nnz, A, rowPtr, colIdx, x, b);
	CUDA_CHECK_ERROR()

	// execute plan
	culaSparseStatus status = culaSparseExecutePlan(handle, plan, &config, &result);
	CUDA_CHECK_ERROR()

	//print if error
	if (culaSparseNoError != status || false)
	{
		char buffer[512];
		culaSparseGetResultString(handle, &result, buffer, 512);
		CUDA_CHECK_ERROR()
		printf("%s\n", buffer);
	}

	//print resulting vector x if needed
	if (false)
	{
		float* h_x = new float[dim];
		cudaMemcpy(h_x, x, dim * sizeof(float), cudaMemcpyDeviceToHost);
		CUDA_CHECK_ERROR()
		printf("X = [");
		for (int i = 0; i < dim; i++)
		{
			printf(" %f ", h_x[i]);
		}
		printf("]\n");
	}
}

