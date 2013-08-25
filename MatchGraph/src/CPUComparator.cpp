/*
 * CPUComparator.cpp
 *
 * This class executes the image comparison on CPU.
 * It serves as anchor for the actual image comparator.
 *
 *  Created on: Jul 13, 2013
 *      Author: Fabian
 */

#include "CPUComparator.h"
#include <stdio.h> //printf
#include "Tester.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

/*
 * Constructor
 */
CPUComparator::CPUComparator()
{
	compCPU = new Comparator_CPU();
}

/*
 * Destructor
 */
CPUComparator::~CPUComparator()
{
}

/*
 * Function to switch off image comparison and just fill the result array randomly.
 */
void CPUComparator::setRandomMode(bool mode)
{
	randomMode = mode;
}

/*
 * Executes the comparison on the given data and stores the results in the given memory location.
 */
void CPUComparator::doComparison(ImageHandler* iHandler, MatrixHandler* T, int* h_idx1, int* h_idx2, int* h_res, int arraySize)
{
	if (!randomMode) //do actual image comparison
	{
		//OpenCV image comparison (serial on CPU)
		printf("Comparing %i images with OpenCV_CPU...\n", arraySize);

		int res; //buffer for the result of a specific image-pair

		//loop through all given image-pairs
		for(int i = 0; i < arraySize && h_idx1[i] < T->getDimension(); i++)
		{
			//printf("Comparing image %i: %s with image %i, %s\n",h_idx1[i],iHandler->getFullImagePath(h_idx1[i]), h_idx2[i], iHandler->getFullImagePath(h_idx2[i]));

			//compare pair
			res = compCPU->compare(iHandler->getFullImagePath(h_idx1[i]), iHandler->getFullImagePath(h_idx2[i]), false, false);

			//store result
			h_res[i] = res;
		}
	}
	else //random mode: no image comparison done
	{
		printf("No image comparison done. Result-array filled randomly.\n");

		//random result
		int rndRes;

		//fill result array randomly with 0 or 1
		for(int i = 0; i < arraySize && h_idx1[i] < T->getDimension(); i++)
		{
			rndRes = rand() % 2;
			h_res[i] = rndRes;
		}
	}
}
