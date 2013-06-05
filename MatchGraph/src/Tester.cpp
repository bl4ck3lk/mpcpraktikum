/*
 * Tester.cpp
 *
 *  Created on: Jun 5, 2013
 *      Author: gufler
 */

#include "Tester.h"
#include "CPUImpl.h"
#include <stdio.h>

void Tester::testLaplacian(char* gpuData, int* laplacian, int dim, float lambda)
{
	CPUImpl* cpuM = new CPUImpl();
	cpuM->init(dim, lambda);
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			cpuM->set(i, j, gpuData[i*dim + j]);
		}
	}
	Eigen::MatrixXf laplacianCPU = cpuM->getModLaplacian();

	bool allCorrect = true;

	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			char cpu_val = laplacianCPU(i,j);
			char gpu_val = laplacian[i*dim + j];
			if(gpu_val != cpu_val)
			{
				printf("Err: Lapliacan differs! (%d, %d) is %d on GPU and %d on CPU\n", i, j, gpu_val, cpu_val);
				allCorrect = false;
			}
		}
	}

	if(allCorrect)
		printf("Laplacian CORRECT!");

}
