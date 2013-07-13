/*
 * InitializerCPU.cpp
 *
 *  Created on: Jul 13, 2013
 *      Author: schwarzk
 */

#include "InitializerCPU.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <map>
#include <set>
#include "Tester.h"
#include "CPUImpl.h"

InitializerCPU::InitializerCPU()
{
}

InitializerCPU::~InitializerCPU()
{
}

/*
 * Initializes the T-Matrix with random image comparisons.
 */
void InitializerCPU::doInitializationPhase(MatrixHandler* T, ImageHandler* iHandler, ImageComparator* comparator, int initArraySize)
{
	bool debugPrint = false;

	unsigned int dim = T->getDimension();
	CPUImpl* T_cpu = dynamic_cast<CPUImpl*> (T);

	int* idx1 = new int[initArraySize];
	int* idx2 = new int[initArraySize];
	int* res = new int[initArraySize];
	memset(idx1, 0, sizeof(idx1));
	memset(idx2, 0, sizeof(idx2));
	memset(res, 0, sizeof(res));

	//1. Generate random index arrays
	std::set<long> initIndices;
	std::multimap<int, int> indicesXY; //sorted by idx1 (multiple keys allowed)

	//random seed
	srand (time(NULL));

	//generate random indices
	long rnd;
	int count = 0;
	do {
		rnd = rand() % (dim*dim);

		if ((rnd >= 1+(rnd/dim)+(rnd/dim)*dim) && (initIndices.find(rnd) == initIndices.end()))
		{
			//rnd within upper diagonal matrix w/o diagonal elements and not yet saved
			initIndices.insert(rnd);
			count++;
		}
	} while (count < initArraySize && count < dim*((dim-1)/2)); //maximal #elements in upper diagonal matrix

	//fill multimap to get it sorted by index 1
	for (std::set<long>::iterator iter = initIndices.begin(); iter != initIndices.end(); ++iter)
	{
		indicesXY.insert(std::pair<int, int>((*iter)/dim, (*iter)%dim));
	}

	//fill index arrays
	count = 0;
	for (std::multimap<int, int>::iterator iter = indicesXY.begin(); iter != indicesXY.end(); ++iter, count++)
	{
		idx1[count] = iter->first;
		idx2[count] = iter->second;
	}

	if (debugPrint)
	{
		Tester::printArray(idx1, initArraySize);
		Tester::printArray(idx2, initArraySize);
	}

	//2. Compare these random images
	comparator->doComparison(iHandler, T, idx1, idx2, res, initArraySize);

	if (debugPrint)
	{
		Tester::printArray(res, initArraySize);
	}

	//3. Update T Matrix
	T_cpu->set(idx1, idx2, res, initArraySize);

	//4. Cleanup
	delete[] idx1;
	delete[] idx2;
	delete[] res;
}
