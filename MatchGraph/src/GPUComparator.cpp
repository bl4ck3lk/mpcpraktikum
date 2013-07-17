/*
 * GPUComparator.cpp
 *
 *  Created on: May 29, 2013
 *      Author: Fabian
 */

#include "GPUComparator.h"
#include <stdio.h> //printf
#include "Helper.h"
#include "Tester.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

GPUComparator::GPUComparator()
{
	openCVcomp = new ComparatorCVGPU();

	h_idx1 = NULL;
	h_idx2 = NULL;
	h_res = NULL;
	currentArraySize = 0;
}

GPUComparator::~GPUComparator()
{
	if (h_idx1 != NULL) free(h_idx1);
	if (h_idx2 != NULL) free(h_idx2);
	if (h_res != NULL) free(h_res);

	delete openCVcomp;
}

void GPUComparator::initArrays(int arraySize)
{
	if (h_idx1 != NULL) free(h_idx1);
	if (h_idx2 != NULL) free(h_idx2);
	if (h_res != NULL) free(h_res);

	h_idx1 = (int*) malloc(arraySize*sizeof(int));
	h_idx2 = (int*) malloc(arraySize*sizeof(int));
	h_res = (int*)  malloc(arraySize*sizeof(int));
}

void GPUComparator::setRandomMode(bool mode)
{
	randomMode = mode;
}

void GPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize)
{
	if (!randomMode)
	{
		//OpenCV Image Comparison
		if (arraySize != currentArraySize) initArrays(arraySize);
		h_idx1 = Helper::downloadGPUArrayInt(d_idx1, arraySize);
		h_idx2 = Helper::downloadGPUArrayInt(d_idx2, arraySize);
		h_res = Helper::downloadGPUArrayInt(d_res, arraySize);

		printf("Comparing %i images with OpenCV_GPU...\n", arraySize);
		openCVcomp->compareGPU(iHandler, h_idx1, h_idx2, h_res, arraySize, false, false);

		//upload Result
		Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);
	}
	else
	{
		printf("No image comparison done. Result-array filled randomly.\n");
		int* h_idx1 = Helper::downloadGPUArrayInt(d_idx1, arraySize);
		int* h_res = Helper::downloadGPUArrayInt(d_res, arraySize);
		//random seed
		//srand (time(NULL));
		int rndRes;

		for(int i = 0; i < arraySize && h_idx1[i] < T->getDimension(); i++)
		{
			rndRes = rand() % 2;
			h_res[i] = rndRes;
		}

		//upload Result
		Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);

		free(h_idx1);
		free(h_res);
	}
}

