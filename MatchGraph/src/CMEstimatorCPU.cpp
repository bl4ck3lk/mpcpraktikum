/*
 * CMEstimatorCPU.cpp
 *
 * CPU implementation of finding the k-best image-pairs. These pairs are
 * stored in the resulting index arrays, managed by this class.
 * The resulting arrays contain only image-pairs that have not yet been
 * compared and they do not contain symmetric entries.
 *
 *  Created on: 29.05.2013
 *      Author: Fabian
 */

#include "CMEstimatorCPU.h"
#include <stdio.h> //printen
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <algorithm> //std::find
#include <iostream>
#include <map>
#include "Tester.h"

/*
 * Constructor
 */
CMEstimatorCPU::CMEstimatorCPU()
{
	idx1 = NULL;
	idx2 = NULL;
	res = NULL;
	currentArraySize = 0;
}

/*
 * Destructor
 */
CMEstimatorCPU::~CMEstimatorCPU()
{
	if (idx1 != NULL) delete[] idx1;
	if (idx2 != NULL) delete[] idx2;
	if (res != NULL) delete[] res;
}

/*
 * Find the k-best image-pairs in the given confidence measure matrix. These image-pairs
 * are stored in the allocated index arrays on host.
 */
void CMEstimatorCPU::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	//printf("Determine kBest confidence measures on CPU:\n");

	bool debugPrint = false;

	int dim = T->getDimension();

	//initialize index arrays
	if (currentArraySize != kBest) initIdxArrays(kBest, dim);

	std::map<float, long> confMeasureWithIndex; //continuous index
	std::multimap<int, int> indices; //sorted by idx1 (multiple keys allowed)

	/*
	 * For loop only traverses the upper diagonal matrix without the diagonal elements.
	 */
	for (long i = 1; i < dim*dim; (((i+1)%dim) == 0) ? i += (((i+1)/dim) + 2) : i++)
	{
		//compute matrix indices with given continuous index sequence
		int x = i/dim;
		int y = i%dim;

		//get information status for this index
		char tval = T->getVal(x, y);

		if (tval == 0) //not yet compared
		{
			float value = F[i];
			confMeasureWithIndex.insert(std::pair<float, long>(value, i));
		}
	}

	int count = 0;
	for (std::map<float, long>::reverse_iterator iter = confMeasureWithIndex.rbegin(); iter != confMeasureWithIndex.rend() && count < kBest; ++iter, count++)
	{
		indices.insert(std::pair<int, int>(iter->second/dim, iter->second%dim));
	}

	if (debugPrint) //debug print
	{
		Tester::printArray(idx1, kBest);
		Tester::printArray(idx2, kBest);
		Tester::printArray(res, kBest);
	}

	//sorted arrays
	count = 0;
	for (std::multimap<int, int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, count++)
	{
		idx1[count] = iter->first;
		idx2[count] = iter->second;
	}

	if (debugPrint) //debug print
	{
		printf("[ESTIMATOR]: sorted.\n");
		Tester::printArray(idx1, kBest);
		Tester::printArray(idx2, kBest);
		Tester::printArray(res, kBest);
	}
}

/*
 * Return pointer to host index array1 (i-th index).
 */
int* CMEstimatorCPU::getIdx1Ptr()
{
	return idx1;
}

/*
 * Return pointer to host index array2 (j-th index).
 */
int* CMEstimatorCPU::getIdx2Ptr()
{
	return idx2;
}

/*
 * Return pointer to host array for image-comparison result.
 */
int* CMEstimatorCPU::getResPtr()
{
	return res;
}

/*
 * Allocate resulting arrays.
 */
void CMEstimatorCPU::initIdxArrays(int arraySize, int dim)
{
	if (idx1 != NULL) delete[] idx1;
	if (idx2 != NULL) delete[] idx2;
	if (res != NULL) delete[] res;

	idx1 = new int[arraySize];
	idx2 = new int[arraySize];
	res = new int[arraySize];

	for(int i = 0; i < arraySize; i++)
	{
		idx1[i] = dim+1;
		idx2[i] = dim+1;
		res[i] = 0;
	}
}

/*
 * Fill index array1 and index array2 with random image pairs for random
 * iteration.
 */
void CMEstimatorCPU::computeRandomComparisons(MatrixHandler* T, const int k)
{
	//todo
	//not yet implemented for CPU-Estimator
}
