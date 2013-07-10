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
}


void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize)
{
	if (false) //todo remove me (just for testing purpose)
	{
		//random seed
		srand (time(NULL));
		int rndRes;

		for(int i = 0; i < arraySize && d_idx1[i] < T->getDimension(); i++)
		{
			rndRes = rand() % 2;
			d_res[i] = rndRes;
		}
	}

	//OpenCV Image Comparison
	int* h_idx1 = new int[arraySize];
	int* h_idx2 = new int[arraySize];
	int* h_res = new int[arraySize];
	h_idx1 = Helper::downloadGPUArrayInt(d_idx1, arraySize);
	h_idx2 = Helper::downloadGPUArrayInt(d_idx2, arraySize);
	h_res = Helper::downloadGPUArrayInt(d_res, arraySize);

	printf("Comparing images with OpenCV_GPU...\n");
	openCVcomp->compareGPU(iHandler, h_idx1, h_idx2, h_res, arraySize, true, false);

	//upload Result
	Helper::cudaMemcpyArrayInt(h_res, d_res, arraySize);

	//cleanup
	delete[] h_idx1;
	delete[] h_idx2;
	delete[] h_res;
}

