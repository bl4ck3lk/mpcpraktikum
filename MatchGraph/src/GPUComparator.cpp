/*
 * GPUComparator.cpp
 *
 * This class executes the image comparison on GPU.
 * It serves as anchor for the actual image comparator.
 *
 *  Created on: May 29, 2013
 *      Author: Fabian, Armin
 */

#include "GPUComparator.h"
#include <stdio.h> //printf
#include "Helper.h"
#include "Tester.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

inline __int64_t continuousTimeNs()
 {
	timespec now;
	clock_gettime(CLOCK_REALTIME, &now);

	__int64_t result = (__int64_t ) now.tv_sec * 1000000000
			+ (__int64_t ) now.tv_nsec;

	return result;
}

/*
 * Constructor
 */
GPUComparator::GPUComparator()
{
	openCVcomp = new ComparatorCVGPU();

	h_idx1 = NULL;
	h_idx2 = NULL;
	h_res = NULL;
	currentArraySize = 0;
	totalTime = 0;
}

/*
 * Destructor
 */
GPUComparator::~GPUComparator()
{
	printf("Total comparison time:%f\n", totalTime*(1/(double)1000000000)); 
 	if (h_idx1 != NULL) free(h_idx1);
 	if (h_idx2 != NULL) free(h_idx2);
 	if (h_res != NULL) free(h_res);

	delete openCVcomp;
}

/*
 * Allocate buffer arrays.
 */
void GPUComparator::initArrays(int arraySize)
{
	if (h_idx1 != NULL) free(h_idx1);
	if (h_idx2 != NULL) free(h_idx2);
	if (h_res != NULL) free(h_res);

	h_idx1 = (int*) malloc(arraySize*sizeof(int));
	h_idx2 = (int*) malloc(arraySize*sizeof(int));
	h_res = (int*)  malloc(arraySize*sizeof(int));
}

/*
 * Function to switch off image comparison and just fill the result array randomly.
 */
void GPUComparator::setRandomMode(bool mode)
{
	randomMode = mode;
}

/*
 * Executes the comparison on the given data and stores the results in the given memory location.
 */
void GPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize)
{
	if (!randomMode) //do actual image comparison with OpenCV
	{
		//init buffer arrays if needed
		if (arraySize != currentArraySize) initArrays(arraySize);
		Helper::cudaMemcpyArrayIntToHost(d_idx1, h_idx1, arraySize);
		Helper::cudaMemcpyArrayIntToHost(d_idx2, h_idx2,  arraySize);
		Helper::cudaMemcpyArrayIntToHost(d_res, h_res, arraySize);

		//start runtime measurement for comparison
		__int64_t startTime = continuousTimeNs();

		//compare all given image-pairs
		openCVcomp->compareGPU(iHandler, h_idx1, h_idx2, h_res, arraySize);

		//stop runtime measurement for comparison
		__int64_t timeNeeded = continuousTimeNs()-startTime;

		//sum up over all match graph iterations
		totalTime += timeNeeded;

		//printf("Compared %i images with OpenCV_GPU. Time: %f\n", arraySize,timeNeeded*(1/(double)1000000000));
		printf("%f\n", timeNeeded*(1/(double)1000000000));

		//upload Result
		Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);
	}
	else //random mode: no image comparison done
	{
		printf("No image comparison done. Result-array filled randomly.\n");

		//download arrays to host, for host access
		int* h_idx1 = Helper::downloadGPUArrayInt(d_idx1, arraySize);
		int* h_res = Helper::downloadGPUArrayInt(d_res, arraySize);

		//random seed
		int rndRes;

		//fill result array randomly with 0 or 1
		for(int i = 0; i < arraySize && h_idx1[i] < T->getDimension(); i++)
		{
			rndRes = rand() % 2;
			h_res[i] = rndRes;
		}

		//upload Result
		Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);

		//free host memory
		free(h_idx1);
		free(h_res);
	}
}

