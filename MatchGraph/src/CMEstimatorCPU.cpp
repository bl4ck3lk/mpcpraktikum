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
#include <iostream> //TODO remove later

CMEstimatorCPU::CMEstimatorCPU() {
	// TODO Auto-generated constructor stub

}

Indices* CMEstimatorCPU::getInitializationIndices(MatrixHandler* T, int initNr)
{
	Indices* initIndices = new Indices[initNr];
	std::vector<int> chosenOnes; //max size will be initNr
	int dim = T->getDimension();

	//generate random index
	srand (time(NULL));
	const int MAX_ITERATIONS = dim*(dim/2) + dim; //#elements in upper diagonal matrix + dim

	//generate initialization indices
	for(int i = 0; i < initNr; i++)
	{
		int rIdx = -1;
		int x, y;

		int c = 0;
		do {
			//get random number
			rIdx = rand() % (dim*dim);

			//compute matrix indices with given continuous index sequence
			x = rIdx/dim;
			y = rIdx%dim;
			c++;
		} while ( ((rIdx < 1+(rIdx/dim)+(rIdx/dim)*dim)
					|| (T->getVal(x,y) != 0)
					|| (std::find(chosenOnes.begin(), chosenOnes.end(), rIdx) != chosenOnes.end()))
				&& (c <= MAX_ITERATIONS) );
		/* :TRICKY:
		 * As long as the random number is not within the upper diagonal matrix w/o diagonal elements
		 * or T(idx) != 0 generate or already in the list of Indices, a new random index but maximal
		 * MAX_ITERAtION times.
		 */

		if (c <= MAX_ITERATIONS) //otherwise initIndices contains -1 per struct definition
		{
			chosenOnes.push_back(rIdx);
			initIndices[i].i = x;
			initIndices[i].j = y;
		}
	}

	return initIndices;
}

Indices* CMEstimatorCPU::getKBestConfMeasures(MatrixHandler* T, float* F, int kBest)
{
	printf("Determine kBest confidence measures on CPU:\n");

	int dim = T->getDimension();
	Entry* kBestEntries = new Entry[kBest];
	Indices* kBestIndices = new Indices[kBest];

	//Initialize list
	for (int j = 0; j < kBest; j++)
	{
		kBestEntries[j].value = -10000.0;
		kBestEntries[j].i = -1;
		kBestEntries[j].j = -1;
	}



	//linear search
	/* :TRICKY:
	 * For loop only traverses the upper diagonal matrix without the diagonal elements.
	 */
	for (int i = 1; i < dim*dim; (((i+1)%dim) == 0) ? i += (((i+1)/dim) + 2) : i++)
	{
		//compute matrix indices with given continuous index sequence
		int x = i/dim;
		int y = i%dim;

		//get information status for this index
		char tval = T->getVal(x, y);
		//std::cout << "tval (" << i/dim << "," << i%dim<< ") = "<< 0+tval << std::endl;

		if (tval == 0)
		{
			float value = F[i];

			int findMin = 0;
			float findMinValue = kBestEntries[0].value;
			for (int j = 0; j < kBest; j++)
			{
				if (findMinValue > kBestEntries[j].value)
				{
					findMinValue = kBestEntries[j].value;
					findMin = j;
				}
			}

			if (value > findMinValue)
			{
				kBestEntries[findMin].value = value;
				kBestEntries[findMin].i = x;
				kBestEntries[findMin].j = y;

				kBestIndices[findMin].i = x;
				kBestIndices[findMin].j = y;
			}
		}
	}

	//print
    printf("%i best entries:\n",kBest);
    for (int k = 0; k < kBest; k++)
    {
        printf("%i: %f at [%i,%i]\n",k,kBestEntries[k].value,kBestEntries[k].i,kBestEntries[k].j);
    }

	return kBestIndices;
}
