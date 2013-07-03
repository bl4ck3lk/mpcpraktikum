/*
 * CPUComparator.cpp
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#include "CPUComparator.h"
#include "Comparator_CVGPU.h"
#include <stdio.h> //printf

//todo remove me (random)
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

CPUComparator::CPUComparator()
{
	// TODO Auto-generated constructor stub

}


void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* d_idx1, int* d_idx2, int* d_res, int arraySize)
{
	if (true) //todo remove me (just for testing purpose)
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
}

void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int kBest, Indices* kBestIndices)
{
	//Comparator comparator;
	/* TODO: Comparator should inherit from ImageComparator.h and implement the static function
	 * virtual void doComparison(ImageHandler* iHandler, MatrixHandler* T, int k, Indices* kBestIndices) = 0;
	 * (just like this CPUComparator)
	 */

	ComparatorCVGPU* comparator = new ComparatorCVGPU();

	for(int i = 0; i < kBest; i++)
	{
		//if(doComparisonOfImages(image[index1[i]] , image[index2[i]]) TELLS Similar)
			//THEN T = 1
		//ELSE T = -1

		int x = kBestIndices[i].i;
		int y = kBestIndices[i].j;

		if (x != -1)
		{
			printf("[CPUComparator]: Comparing image %i: %s with image %i: %s.\n", x, iHandler->getFullImagePath(x), y, iHandler->getFullImagePath(y));
			int resultCompare = comparator->compareGPU(iHandler->getFullImagePath(x), iHandler->getFullImagePath(y), false, true);
			bool result = (resultCompare == 1) ? true : false;
			printf(" Result: %i\n", result);

			T->set(x, y, result);
			T->set(y, x, result); //set T symmetrically
		}
	}

	//
}

