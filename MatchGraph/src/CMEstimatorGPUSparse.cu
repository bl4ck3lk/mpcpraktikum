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

const int THREADS = 128;

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

CMEstimatorGPUSparse::CMEstimatorGPUSparse() {
}

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
Indices* CMEstimatorGPUSparse::getKBestConfMeasures(float* xColumnDevice, float* bColumnDevice, int columnIdx, int dim, int kBest)
{
	//storage for the kBest indices
	Indices* kBestIndices = new Indices[kBest];

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
	long* indices = new long[kBest]; //at most kBest indices are needed
	//the first kBest indices are also the best conf. measure values after sorting
	thrust::copy(dp_gpuIndices, dp_gpuIndices + kBest, indices);
	CUDA_CHECK_ERROR();

	//free memory
	cudaFree(gpuIndices);

	//build indices list structure
	for(int i = 0; i<kBest; i++)
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


Indices* CMEstimatorGPUSparse::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("Determine kBest confidence measures on GPU (column-wise):\n");
	//invoked only on sparse matrixhandler
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);

	//indices cache
	Indices* bestIndices = new Indices[kBest];
	int countIndices = 0;

	//set up data for solver
	unsigned int dim = T_sparse->getDimension();
	unsigned int nnz = T_sparse->getNNZ();

	float* d_values = NULL;
	int* d_colIdx = NULL;
	int* d_rowPtr = NULL;

	//x-vector
	float* d_x;
	cudaMalloc((void**) &d_x, dim * sizeof(float));

	//b-vector
	float* d_b;
	cudaMalloc((void**) &d_b, dim * sizeof(float));

	//*****************************************************
	// TODO directly obtain device pointers from GPUSparseB
	d_values = T_sparse->getValueArr(true, NULL, NULL, NULL);

	int* colIdx = T_sparse->getColIdx();
	cudaMalloc((void**) &d_colIdx, nnz * sizeof(int));
	cudaMemcpy(d_colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);

	int* rowPtr = T_sparse->getRowPtr();
	cudaMalloc((void**) &d_rowPtr, (dim+1) * sizeof(int));
	cudaMemcpy(d_rowPtr, rowPtr, (dim+1) * sizeof(int), cudaMemcpyHostToDevice);
	// END *************************************************

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
	for(int i = 0; i < dim && countIndices < kBest; i++) //if enough values are gathered, stop computation
	{
		//0. determine number of best values for this column
		//The bigger i, the less best indices are determined for this column
		int xBestForThisColumn = ((dim-i)/(0.5*dim*(dim-1))) * kBest;
		if (!xBestForThisColumn) xBestForThisColumn = 1; //at least 1 per column
		//take into account that probably not as many indices as needed can be determined, so try o get them in the next column
		int determineXforThisColumn = xBestForThisColumn + (determinedIndicesByNow - countIndices);
		Indices* tmpIndices = new Indices[determineXforThisColumn];

		//1. Compute confidence measure for this column (solve Ax=b)
		cudaMemcpy(d_b, T_sparse->getColumn(i), dim * sizeof(float), cudaMemcpyHostToDevice);
		computeConfidenceMeasure(handle, plan, config, dim, nnz, d_values, d_rowPtr, d_colIdx, d_x, d_b);

		//2. get indices of x best confidence measure values
		tmpIndices = getKBestConfMeasures(d_x, d_b, i, dim, determineXforThisColumn);

		//3. gather indices
		for(int j = 0; j < determineXforThisColumn && countIndices < kBest; j++)
		{
			if (-1 == tmpIndices[j].i) break; //following indices are also -1
			else
			{
				bestIndices[countIndices] = tmpIndices[j];
				countIndices++;
			}
		}

		determinedIndicesByNow += xBestForThisColumn; // #indices that should have been determined

		printf("Column %i, try to determine %i best values. Actually determined by now %i values\n", i, determineXforThisColumn, countIndices);
	}

	//clean up the mess
	cudaFree(d_x);
	cudaFree(d_b);
	culaSparseDestroyPlan(plan);
	culaSparseDestroy(handle);

	//print
	if (true)
	{
		printf("%i best entries:\n", kBest);
		for(int i = 0; i < kBest; i++)
		{
			if (bestIndices[i].i != -1)
			{
				//value can't be printed because it is not saved in the Indices-list
				printf("%i: at [%i,%i]\n",i,bestIndices[i].i,bestIndices[i].j);
			}
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

	// execute plan
	culaSparseStatus status = culaSparseExecutePlan(handle, plan, &config, &result);

	//print if error
	if (culaSparseNoError != status || false)
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
		printf("[");
		for (int i = 0; i < dim; i++)
		{
			printf(" %f ", h_x[i]);
		}
		printf("]\n");
	}
}

