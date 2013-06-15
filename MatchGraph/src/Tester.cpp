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
	CPUImpl* cpuM = new CPUImpl(dim, lambda);
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			cpuM->set(i, j, gpuData[i * dim + j]);
		}
	}
	Eigen::MatrixXf laplacianCPU = cpuM->getModLaplacian();

	bool allCorrect = true;

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			char cpu_val = laplacianCPU(i, j);
			char gpu_val = laplacian[i * dim + j];
			if (gpu_val != cpu_val)
			{
				printf("Err: Lapliacan differs! (%d, %d) is %d on GPU and %d on CPU\n", i, j, gpu_val, cpu_val);
				allCorrect = false;
			}
		}
	}

	if (allCorrect)
		printf("Laplacian CORRECT!");

}

void Tester::printArrayInt(int* arr, int n)
{
	printf("[");
	for (int i = 0; i < n; i++)
	{
		if(i == n-1)
			printf(" %d ", arr[i]);
		else
			printf(" %d, ", arr[i]);
	}
	printf("]\n");
}

void Tester::printMatrixArrayInt(int* matrArr, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{

			int val = matrArr[i * dim + j];
			if (val < 0)
			{
				printf("  %d ", val);
			}
			else
			{
				printf("   %d ", val);
			}

		}
		printf("\n");
	}
	printf("\n");
}

void Tester::printArrayChar(char* arr, int n)
{
	printf("[");
	for (int i = 0; i < n; i++)
	{
		if(i == n-1)
			printf(" %d ", arr[i]);
		else
			printf(" %d, ", arr[i]);
	}
	printf("]\n");
}

void Tester::printMatrixArrayChar(char* matrArr, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{

			char val = matrArr[i * dim + j];
			if (val < 0)
			{
				printf("  %d ", val);
			}
			else
			{
				printf("   %d ", val);
			}

		}
		printf("\n");
	}
	printf("\n");
}

void Tester::printArrayFloat(float* arr, int n)
{
	printf("[");
	for (int i = 0; i < n; i++)
	{
		if(i == n-1)
			printf(" %f ", arr[i]);
		else
			printf(" %f, ", arr[i]);
	}
	printf("]\n");
}
