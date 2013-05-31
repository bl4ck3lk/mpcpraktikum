/*
 * CPUComparator.cpp
 *
 *  Created on: May 29, 2013
 *      Author: gufler
 */

#include "CPUComparator.h"
#include <stdio.h> //printf

CPUComparator::CPUComparator()
{
	// TODO Auto-generated constructor stub

}

void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int kBest, Indices* kBestIndices)
{
	//printf("Executing Comparison on CPU.\n");
	for(int i = 0; i < kBest; i++)
	{
		//if(doComparisonOfImages(image[index1[i]] , image[index2[i]]) TELLS Similar)
			//THEN T = 1
		//ELSE T = -1

		int x = kBestIndices[i].i;
		int y = kBestIndices[i].j;

		if (x != -1)
		{

			//printf("Comparator: Comparing image %i: %s and image %i: %s \n",x,y,iHandler->getImage(x),iHandler->getImage(y));
			//map has lost elements, so this is not working yet

			T->set(x, y, 1.0);
			T->set(y, x, 1.0); //set T symmetrically
		}
	}

	//
}

