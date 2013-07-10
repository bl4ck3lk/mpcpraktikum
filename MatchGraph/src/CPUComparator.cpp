/*
 * CPUComparator.cpp
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#include "CPUComparator.h"
#include <stdio.h> //printf
#include "Helper.h"
#include "Tester.h"

//todo remove me (random)
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

CPUComparator::CPUComparator()
{
	openCVcomp = new ComparatorCVGPU();

	h_idx1 = NULL;
	h_idx2 = NULL;
	h_res = NULL;
	currentArraySize = 0;
}

CPUComparator::~CPUComparator()
{
	if (h_idx1 != NULL) delete[] h_idx1;
	if (h_idx2 != NULL) delete[] h_idx2;
	if (h_res != NULL) delete[] h_res;
}

void CPUComparator::initArrays(int arraySize)
{
	if (h_idx1 != NULL) delete[] h_idx1;
	if (h_idx2 != NULL) delete[] h_idx2;
	if (h_res != NULL) delete[] h_res;

	h_idx1 = new int[arraySize];
	h_idx2 = new int[arraySize];
	h_res = new int[arraySize];
}

void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize)
{
	bool imageComparison = true;

	if (imageComparison)
	{
		//OpenCV Image Comparison
		if (arraySize != currentArraySize) initArrays(arraySize);
		h_idx1 = Helper::downloadGPUArrayInt(d_idx1, arraySize);
		h_idx2 = Helper::downloadGPUArrayInt(d_idx2, arraySize);
		h_res = Helper::downloadGPUArrayInt(d_res, arraySize);

		printf("Comparing images with OpenCV_GPU...\n");
		openCVcomp->compareGPU(iHandler, h_idx1, h_idx2, h_res, arraySize, true, false);

		//upload Result
		Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);
	}
	else
	{
		printf("No image comparison done. Result-array filled randomly.\n");
		int* h_idx1 = Helper::downloadGPUArrayInt(d_idx1, arraySize);
		int* h_res = Helper::downloadGPUArrayInt(d_res, arraySize);
		//random seed
		srand (time(NULL));
		int rndRes;

		for(int i = 0; i < arraySize && h_idx1[i] < T->getDimension(); i++)
		{
			rndRes = rand() % 2;
			h_res[i] = rndRes;
		}

		//upload Result
		Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);

		delete[] h_idx1;
		delete[] h_res;
	}
}

