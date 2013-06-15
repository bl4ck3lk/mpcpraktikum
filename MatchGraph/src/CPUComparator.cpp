/*
 * CPUComparator.cpp
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#include "CPUComparator.h"
//#include "Comparator.h"
#include <stdio.h> //printf

CPUComparator::CPUComparator()
{
	// TODO Auto-generated constructor stub

}

void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int kBest, Indices* kBestIndices)
{
	//Comparator comparator;
	/* TODO: Comparator should inherit from ImageComparator.h and implement the static function
	 * virtual void doComparison(ImageHandler* iHandler, MatrixHandler* T, int k, Indices* kBestIndices) = 0;
	 * (just like this CPUComparator)
	 */

	for(int i = 0; i < kBest; i++)
	{
		//if(doComparisonOfImages(image[index1[i]] , image[index2[i]]) TELLS Similar)
			//THEN T = 1
		//ELSE T = -1

		int x = kBestIndices[i].i;
		int y = kBestIndices[i].j;

		if (x != -1)
		{
			printf("[CPUComparator]: Comparing image %i: %s with image %i: %s\n", x, iHandler->getFullImagePath(x), y, iHandler->getFullImagePath(y));


			T->set(x, y, 1.0);
			T->set(y, x, 1.0); //set T symmetrically
		}
	}

	//
}

