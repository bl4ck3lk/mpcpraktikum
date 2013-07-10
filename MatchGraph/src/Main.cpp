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
#include "CMEstimatorGPUSparse.h"
#include "Initializer.h"
#include "ImageHandler.h"
#include "Tester.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>

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
	MatrixHandler* T;
	GPUSparse* T_sparse;
	Initializer* init = new Initializer();
	//CMEstimator* CME = new CMEstimatorCPU();
    //CMEstimator* CME = new CMEstimatorGPUSorted();
	//CMEstimator* CME = new CMEstimatorGPUApprox();
	CMEstimator* CME = new CMEstimatorGPUSparse();
	CMEstimatorGPUSparse* CME_sparse = dynamic_cast<CMEstimatorGPUSparse*> (CME);
	ImageHandler* iHandler = new ImageHandler(dir, imgExension);
	ImageComparator* comparator = new CPUComparator();


//	iHandler->fillWithEmptyImages(5000); //todo remove me. for testing purpose
	printf("Directory %s with %i files initialized.\n", dir, iHandler->getTotalNr());

	////////////
	//Settings//
	////////////
	int dim			= iHandler->getTotalNr();

	int sizeOfInitIndicesList 		= 30; //todo make this dependent from dim (exponential)

	float lambda 	= 1.0;
	int iterations 	= 4;
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
			printf("Initializing Matrix.\n");
			T = new GPUSparse(dim, lambda); //empty Matrix (test = false)
			T_sparse = dynamic_cast<GPUSparse*> (T);
			init->doInitializationPhase(T, iHandler, comparator, sizeOfInitIndicesList);
			printf("Initialization of T done. \n");
		}

		/////////////////////////
		//Iterative progression//
		/////////////////////////
		printf("************** Iteration %i **************\n", i);

		//get the next images that should be compared (implicitly solving eq. system => confidence measure matrix)
		CME_sparse->getKBestConfMeasuresSparse(T, NULL, kBest);

		//compare images which are located in the device arrays of CME_sparse
		comparator->doComparison(iHandler, T, CME_sparse->getIdx1DevicePtr(), CME_sparse->getIdx2DevicePtr(), CME_sparse->getResDevicePtr(), kBest);

		//update matrix with new information (compared images)
		T_sparse->updateSparseStatus(CME_sparse->getIdx1DevicePtr(), CME_sparse->getIdx2DevicePtr(), CME_sparse->getResDevicePtr(), kBest);
	}

	printf("Resulting Matrix:\n");
	T->print();

	//cleanup
	delete T_sparse;
	delete CME_sparse;

	return 0;
}
