/*
 * CMEstimatorCPU.cpp
 *
 *  Created on: 29.05.2013
 *      Author: furby
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

CMEstimatorCPU::CMEstimatorCPU()
{
	idx1 = NULL;
	idx2 = NULL;
	res = NULL;
	currentArraySize = 0;
}

CMEstimatorCPU::~CMEstimatorCPU()
{
	//todo
}

void CMEstimatorCPU::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("Determine kBest confidence measures on CPU:\n");
	int dim = T->getDimension();

	//initialize index arrays
	if (currentArraySize != kBest) initIdxArrays(kBest, dim);

	std::map<float, long> confMeasureWithIndex;

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

		if (tval == 0)
		{
			float value = F[i];
			confMeasureWithIndex.insert(std::pair<float, long>(value, i));
		}
	}

	int count = 0;
	for (std::map<float, long>::reverse_iterator iter = confMeasureWithIndex.rbegin(); iter != confMeasureWithIndex.rend() && count < kBest; ++iter)
	{
		idx1[count] = iter->second/dim;
		idx2[count] = iter->second%dim;
		count++;
	}

	if (true) //debug print
	{
		Tester::printArrayInt(idx1, kBest);
		Tester::printArrayInt(idx2, kBest);
		Tester::printArrayInt(res, kBest);
	}
}

int* CMEstimatorCPU::getIdx1Ptr()
{
	return idx1;
}

int* CMEstimatorCPU::getIdx2Ptr()
{
	return idx2;
}

int* CMEstimatorCPU::getResPtr()
{
	return res;
}

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
