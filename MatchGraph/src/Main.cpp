/*
 * Main.cpp
 *
 *  Created on: May 29, 2013
 *      Author: Fabian, gufler
 */

#include "GPUSparse.h"
#include "GPUMatrix.h"
#include "CPUImpl.h"
#include "GPUComparator.h"
#include "CPUComparator.h"
#include "CMEstimatorCPU.h"
#include "CMEstimatorGPUSparse.h"
#include "Initializer.h"
#include "InitializerGPU.h"
#include "InitializerCPU.h"
#include "ImageHandler.h"
#include "Tester.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>

#define GPU_VERSION 1

int main(int argc, char** argv)
{
	//const char* dir = "../resource/notre_dame_40"; //TODO use input-parameter
	//const char* dir = "/graphics/projects/data/photoDB_fromWWW/photoCollections/Flickr/A/aachen";
	//const char* imgExension = ".jpg";
	const char* dir = "/graphics/projects/data/canon_5d/2012_12_11_similar_pics";
	const char* imgExension = ".ppm";

	//Initialize Cuda device
	findCudaDevice(argc, (const char **)argv);

	////////////////////////
	//Computation handlers//
	////////////////////////
	ImageHandler* iHandler = new ImageHandler(dir, imgExension);
	MatrixHandler* T;
	CMEstimator* CME;
	Initializer* init;
	ImageComparator* comparator;

//	iHandler->fillWithEmptyImages(30); //todo remove me. for testing purpose
	printf("Directory %s with %i files initialized.\n", dir, iHandler->getTotalNr());

	////////////
	//Settings//
	////////////
	int dim			= iHandler->getTotalNr();
	float lambda 	= 1.0;
	int iterations 	= 4;
	int sizeOfInitIndicesList = 50; //todo make this dependent from dim (exponential)
	int kBest 		= sizeOfInitIndicesList;

	////////////////////////////////////
	// Initialize computation handler //
	////////////////////////////////////
#if GPU_VERSION
	printf("Executing GPU Version.\n");
	T = new GPUSparse(dim, lambda);
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*> (T);
	init = new InitializerGPU();
	CME = new CMEstimatorGPUSparse();
	comparator = new GPUComparator();
#else
	printf("Executing CPU Version.\n");
	T = new CPUImpl(dim, lambda);
	CPUImpl* T_cpu = dynamic_cast<CPUImpl*> (T);
	init = new InitializerCPU();
	CME = new CMEstimatorCPU();
	comparator = new CPUComparator();
#endif

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
			printf("Initializing Matrix.\n");
			init->doInitializationPhase(T, iHandler, comparator, sizeOfInitIndicesList);
			printf("Initialization of T done. \n");
		}

		/////////////////////////
		//Iterative progression//
		/////////////////////////
		printf("************** Iteration %i **************\n", i);

#if GPU_VERSION
		//get the next images that should be compared (implicitly solving eq. system => confidence measure matrix)
		CME->getKBestConfMeasures(T, NULL, kBest);
		//compare images which are located in the device arrays of CME
		comparator->doComparison(iHandler, T, CME->getIdx1Ptr(), CME->getIdx2Ptr(), CME->getResPtr(), kBest); //device pointer
		//update matrix with new information (compared images)
		T_sparse->updateSparseStatus(CME->getIdx1Ptr(), CME->getIdx2Ptr(), CME->getResPtr(), kBest); //device pointer
#else
		//CPU Version
		//get the next images that should be compared
		CME->getKBestConfMeasures(T, T->getConfMatrixF(), kBest);
		//compare images which are located in the arrays of CME
		comparator->doComparison(iHandler, T, CME->getIdx1Ptr(), CME->getIdx2Ptr(), CME->getResPtr(), kBest); //host pointer
		//update matrix with new information (compared images)
		T_cpu->set(CME->getIdx1Ptr(), CME->getIdx2Ptr(), CME->getResPtr(), kBest); //host pointer
#endif
	}

	printf("Resulting Matrix:\n");
	T->print();

	//cleanup
#if GPU_VERSION
	delete T_sparse;
#else

#endif
	delete CME;

	return 0;
}
