/*
 * Main.cpp
 *
 *  Created on: May 29, 2013
 *      Author: gufler, Fabian
 */

#include "GPUSparse.h"
#include "GPUMatrix.h"
#include "CPUImpl.h"
#include "CPUComparator.h"
#include "CMEstimatorCPU.h"
#include "CMEstimatorGPUSorted.h"
#include "CMEstimatorGPUApprox.h"
#include "ImageHandler.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cula.h>

#define TESTMATRIX 0 //to enable test matrix, one has to set the test bool in T->init to true

int main(int argc, char** argv)
{
	const char* dir = "../resource/notre_dame_40"; //TODO use input-parameter

	////////////////////////
	//computation handlers//
	////////////////////////
	MatrixHandler* T;
	CMEstimator* CME = new CMEstimatorCPU();
    	//CMEstimator* CME = new CMEstimatorGPUSorted();
    	//CMEstimator* CME = new CMEstimatorGPUApprox();
	ImageComparator* comparator = new CPUComparator();
	ImageHandler* iHandler = new ImageHandler(dir);

	//iHandler->fillWithEmptyImages(10);

	printf("Directory %s with %i files initialized.\n", dir, iHandler->getTotalNr());

	////////////
	//Settings//
	////////////
	int dim			= iHandler->getTotalNr();
#if TESTMATRIX
	dim	 		= 10;
#endif

	const int MAX_INIT_ITERATIONS 	= (dim*(dim-1))/2; //#elements in upper diagonal matrix
	const int MIN_INIT_SIMILARITIES = 2*dim;
	int sizeOfInitIndicesList 		= 3;

	float lambda 	= 1.0;
	int iterations 	= 5;
	int kBest 		= sizeOfInitIndicesList;

	/////////////////////////////////////////////////////////
	//Match Graph algorithm (predict & verify step-by-step)//
	/////////////////////////////////////////////////////////
	for(int i = 0; i < iterations; i++)
	{
		if (i == 0)
		{
			////////////////////
			//Initialize Phase//
			////////////////////
			T = new CPUImpl(dim, lambda); //empty Matrix (test = false)
			std::cout << "Init T:\n"<< std::endl;
			T->print();

#if !TESTMATRIX
			int c = 0;
			//Initialization matrix should contain sufficient similarities
			while(MIN_INIT_SIMILARITIES >  T->getSimilarities()  && MAX_INIT_ITERATIONS > c)
			{
				printf("similar %d \n", T->getSimilarities());

				//get random indices for initialization
				Indices* initIndices = CME->getInitializationIndices(T, sizeOfInitIndicesList);

				//compare these images
				comparator->doComparison(iHandler, T, sizeOfInitIndicesList, initIndices);

				c++;
			}

			if (c != MAX_INIT_ITERATIONS) printf("Enough similarities found.\n");
			else printf("Maximum initialization iterations reached w/o enough similarities.\n");
#endif
			printf("Initialization complete. T:\n");
			T->print();
		}

		/////////////////////////
		//Iterative progression//
		/////////////////////////

		//compute confidence measure matrix
		float* f = T->getConfMatrixF();

		//return 1;

		//determine the k-best values in confidence measure matrix
		Indices* bestIndices = CME->getKBestConfMeasures(T, f, kBest);

		//compare k-best image pairs and update T-matrix respectively
	    comparator->doComparison(iHandler, T, kBest, bestIndices);

		std::cout << "T_" << i << ":\n" << std::endl;
		T->print();
	}
	return 0;
}
