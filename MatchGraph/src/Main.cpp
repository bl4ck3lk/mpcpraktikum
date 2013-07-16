/*
 * Main.cpp
 *
 *	Start class for the match graph algorithm
 *
 *  Created on: May 29, 2013
 *      Author: Fabian, Armin, Julio
 */

#include "GPUSparse.h"
#include "CPUImpl.h"
#include "GPUComparator.h"
#include "CPUComparator.h"
#include "CMEstimatorCPU.h"
#include "CMEstimatorGPUSparse.h"
#include "CMEstimatorGPUSparseMax.h"
#include "Initializer.h"
#include "InitializerGPU.h"
#include "InitializerCPU.h"
#include "ImageHandler.h"
#include "Tester.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <string.h>

#define GPU_VERSION 1

int main(int argc, char** argv)
{
	srand((unsigned int)time(NULL));

	char* usageBuff = new char[1024];
	strcpy(usageBuff, "Usage: <path> <ext> <iter> [<k>] [<lambda>]\n"
			"       Starts algorithm for <iter> iterations on images in directory <path> with specified file\n"
			"       extension <ext>. Parameter <k> [1,#numImages] defines how many images shall be compared each\n"
			"       iteration (k-best). Model parameter lambda [0,1] (default = 1) influences the computation of\n"
			"       confidence measures (see algorithm details)."
			"\nOR\n"
			"       -r <dim> <k> <iter> [<lambda>]\tRandom mode which does comparison not on real image data\n"
			"       but just in a random fashion for dimension <dim> and <iter> iterations and given\n"
			"       parameter <k> [1,dim] and model parameter lambda [0,1] (default = 1)\n");

	if (argc < 4)
	{
		printf("%s", usageBuff);
		exit(EXIT_FAILURE);
	}

	int acount = 1;

	ImageHandler* iHandler; //handler object for image files
	bool randomMode = false;
	const char* dir = argv[acount++];

	int _dim = -1;
	int _iter = 1;
	int _k = 1;
	float _lambda = 1.0;

	if (strcmp(dir, "-r") == 0)
	{
		if (argc < 5)
		{
			printf("%s", usageBuff);
			exit(EXIT_FAILURE);
		}

		randomMode = true;

		_dim = atoi(argv[acount++]);
		if (_dim < 2)
		{
			printf("ERROR: dimension must at least be 2\n");
			exit(EXIT_FAILURE);
		}

		_k = atoi(argv[acount++]);
		if (_k > _dim || _k < 1)
		{
			printf("ERROR: <k> must be >=1 and <= dimension\n");
		}

		_iter = atoi(argv[acount++]);

		iHandler = new ImageHandler("", "");
		iHandler->fillWithEmptyImages(_dim);

		if (argc > 5)
		{
			_lambda = atof(argv[acount++]);
		}
	}
	else
	{
		const char* imgExtension = argv[acount++];
		iHandler = new ImageHandler(dir, imgExtension);
		if (iHandler->getTotalNr() < 2)
		{
			printf("ERROR: no images in given path or invalid path name or invalid file extension\n");
			exit(EXIT_FAILURE);
		}

		_iter = atoi(argv[acount++]);

		if (argc > 4) //optional param k seems to be present
		{
			_k = atoi(argv[acount++]);
		}
		else
		{
			_k = iHandler->getTotalNr() / 2;
		}

		if (argc > 5) //optional param lambda seems to be present
		{
			_lambda = atof(argv[acount++]);
		}

	}
	if (_iter < 1)
	{
		printf("ERROR: iteration number has to be >= 1");
		exit(EXIT_FAILURE);
	}
	if (_lambda < 0 || _lambda > 1)
	{
		printf("ERROR: lambda must be a floating point number in [0,1]\n");
		exit(EXIT_FAILURE);
	}

	/* Some directories...
	 /graphics/projects/data/photoDB_fromWWW/photoCollections/Flickr/A/aachen
	 /graphics/projects/data/canon_5d/2012_12_11_similar_pics
	 * */

	//Initialize Cuda device
	findCudaDevice(argc, (const char **) argv);

	////////////////////////
	//Computation handlers//
	////////////////////////
	MatrixHandler* T;
	CMEstimator* CME;
	Initializer* init;
	ImageComparator* comparator;

	if (!randomMode)
		printf("Directory %s with %i files initialized.\n", dir, iHandler->getTotalNr());

	//////////////////
	//Final Settings//
	//////////////////
	const int dim = iHandler->getTotalNr();
	const int iterations = _iter;
	const float lambda = _lambda;
	const int kBest = _k;
	const int sizeOfInitIndicesList = kBest; //TODO make this dependent from dim (exponential)?

	////////////////////////////////////
	// Initialize computation handler //
	////////////////////////////////////
#if GPU_VERSION
	printf("Executing GPU Version.\n");
	T = new GPUSparse(dim, lambda);
	GPUSparse* T_sparse = dynamic_cast<GPUSparse*>(T);
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

	comparator->setRandomMode(randomMode);

	/////////////////////////////////////////////////////////
	//Match Graph algorithm (predict & verify step-by-step)//
	/////////////////////////////////////////////////////////
	for (int i = 0; i < iterations; i++)
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
//		if(i < 0)
//			CME->computeRandomComparisons(T, kBest);
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
		comparator->doComparison(iHandler, T, CME->getIdx1Ptr(), CME->getIdx2Ptr(), CME->getResPtr(), kBest);//host pointer
		//update matrix with new information (compared images)
		T_cpu->set(CME->getIdx1Ptr(), CME->getIdx2Ptr(), CME->getResPtr(), kBest);//host pointer
#endif
	}

	T_sparse->logSimilarToFile("log/matchGraph.log", iHandler);

	//cleanup
#if GPU_VERSION
	delete T_sparse;
#else

#endif
	delete init;
	delete[] usageBuff;
	delete CME;
	delete comparator;

	exit(EXIT_SUCCESS);
}
