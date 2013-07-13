/*
 * CPUComparator.cpp
 *
 *  Created on: Jul 13, 2013
 *      Author: schwarzk
 */

#include "CPUComparator.h"
#include <stdio.h> //printf
#include "Tester.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

CPUComparator::CPUComparator()
{
	compCPU = new Comparator_CPU();
}

CPUComparator::~CPUComparator()
{
}

void CPUComparator::setRandomMode(bool mode)
{
	randomMode = mode;
}

void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* h_idx1, int* h_idx2, int* h_res, int arraySize)
{
	if (!randomMode)
	{
		unsigned int dim = T->getDimension();

		//OpenCV Image Comparison (on CPU seriell)
		printf("Comparing %i images with OpenCV_CPU...\n", arraySize);
		int res;
		for(int i = 0; i < arraySize && h_idx1[i] < (dim+1); i++)
		{
//			printf("Comparing image %i: %s with image %i, %s\n",h_idx1[i],iHandler->getFullImagePath(h_idx1[i]), h_idx2[i], iHandler->getFullImagePath(h_idx2[i]));
			res = compCPU->compare(iHandler->getFullImagePath(h_idx1[i]), iHandler->getFullImagePath(h_idx2[i]), false, false);
			h_res[i] = res;
		}
	}
	else //random mode: no image comparison for testing purpose
	{
		printf("No image comparison done. Result-array filled randomly.\n");
		//random seed
		srand (time(NULL));
		int rndRes;

		for(int i = 0; i < arraySize && h_idx1[i] < T->getDimension(); i++)
		{
			rndRes = rand() % 2;
			h_res[i] = rndRes;
		}
	}
}
