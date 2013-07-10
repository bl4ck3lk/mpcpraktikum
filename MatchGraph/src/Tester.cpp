/*
 * Tester.cpp
 *
 *  Created on: Jun 5, 2013
 *      Author: gufler
 */

#include "CPUImpl.h"
#include "Tester.h"
#include <stdio.h>
#include <iostream>
#include "GPUSparse.h"

Eigen::MatrixXd assemblyMatrixFromCSR(int* colIdx, int* rowPtr, int* degrees, int dim, double* values)
{
	Eigen::MatrixXd cpuMatrix = Eigen::MatrixXd::Zero(dim, dim);

	for(int r = 0; r < dim; r++	)
	{
		const int start = rowPtr[r];
		const int end = rowPtr[r+1];

		for(int i = start; i < end; i++)
		{

			if(values == NULL)
			{
				cpuMatrix(r, colIdx[i]) = 1.0;
			}
			else
			{
				cpuMatrix(r, colIdx[i]) = values[i];
			}
		}

//		std::set<int>::const_iterator it = dissimilarMap.find(r);
//
//		if(it != dissimilarMap.end())
//		{
//			for(it = dissimilarMap.begin(); it < dissimilarMap.end(); it++)
//			{
//				const int j = *it;
//				cpuMatrix(r, j) = -1;
//			}
//		}
	}

	return cpuMatrix;

}


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

void Tester::printArrayLong(long* arr, int n)
{
	printf("[");
	for (int i = 0; i < n; i++)
	{
		if(i == n-1)
			printf(" %ld ", arr[i]);
		else
			printf(" %ld, ", arr[i]);
	}
	printf("]\n");
}

void Tester::printArrayDouble(double* arr, int n)
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


void Tester::testCSRMatrixUpdate(int* origRowPtr, int* origColIdx, int* degrees,
		int* newRowPtr, int* newColIdx, int* idx1, int* idx2, int* res,  std::map<int,std::set<int> > dissimilarMap, int dim, int k)
{

	Eigen::MatrixXd origCPU = assemblyMatrixFromCSR(origColIdx, origRowPtr, degrees, dim, NULL);

	//do update on cpu
	for(int i = 0; i < k; i++)
	{
		const int x = idx1[i];
		const int y = idx2[i];
		const int val = res[i];

		if(val == 1 && x < dim && y < dim)
		{
			origCPU(x, y) = 1;
			origCPU(y, x) = 1;
		}
	}

	Eigen::MatrixXd updateCPU = assemblyMatrixFromCSR(newColIdx, newRowPtr, degrees, dim, NULL);

	//Compare
	for(int i = 0; i < dim; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			const int valCPU = origCPU(i,j);
			const int valGPU = updateCPU(i,j);
			if(valCPU != valGPU)
			{
				printf("\t ++++++++++++++++++++++++++++++++++ !!! ERROR on UPDATE\n");
				printf("(%i, %i): %i [CPU] vs %i [GPU] \n", i, j, valCPU, valGPU);
				exit(1);
			}
		}
	}

}

void Tester::testValueArray(int* rowPtr, int* colIdx, int* degrees, int dim, int nnz, int lambda, double* gpuResult)
{
	Eigen::MatrixXd cpuMatr = assemblyMatrixFromCSR(colIdx, rowPtr, degrees, dim, NULL);

	double* cpuValueArray = (double*) malloc(nnz * sizeof(double));

	int pos = 0;
	for(int r = 0; r < dim; r++	)
		{
			const int start = rowPtr[r];
			const int end = rowPtr[r+1];

			for(int i = start; i < end; i++, pos++)
			{

				if(colIdx[i] == r)
				{
					cpuValueArray[pos] = 1 + (lambda * dim * degrees[r]);
				}
				else
				{
					cpuValueArray[pos] = -(lambda * dim);
				}
			}
		}

	for(int i = 0; i < nnz; i++)
	{
		const double gpuVal = gpuResult[i];
		const double cpuVal = cpuValueArray[i];
		if(cpuVal != gpuVal)
		{
			GPUSparse::printGpuArray(rowPtr, dim+1, "RowPtr:");

			GPUSparse::printGpuArray(colIdx, dim+1, "ColIdx:");

			printf("\t ++++++++++++++======[VALUE ARRAY] ERROR\n");
			printf("%i: %f [CPU] vs %f [GPU] \n", i, cpuVal, gpuVal);
			exit(1);
		}
	}


	free(gpuResult);

}

void Tester::testColumn(std::map<int,std::set<int> > dissimilarMap, int* rowPtr, int* colIdx, int column, int dim, double* gpuResult)
{
	double* cpuColumn = (double*) malloc(dim*sizeof(double));
	std::fill_n(cpuColumn, dim, 0.0f);

	std::map<int,std::set<int> >::const_iterator it = dissimilarMap.find(column);

	if (it != dissimilarMap.end())
	{
		std::set<int> dis = it->second;
		for (std::set<int>::const_iterator lIter = dis.begin(); lIter != dis.end(); ++lIter)
		{
			int idx = (*lIter);
			cpuColumn[idx] = -1.0f;
		}
	}

	for(int s = rowPtr[column]; s < rowPtr[column+1]; s++)
	{
		int colIdxVal = colIdx[s];
		if(colIdxVal != column)
			cpuColumn[colIdxVal] = 1.0f;
	}

	for(int i = 0; i < dim; i++)
	{
		const double gpuVal = gpuResult[i];
		const double cpuVal = cpuColumn[i];
		if(cpuVal != gpuVal)
		{
			printf("\t ++++++++++++++======[COLUMN %i] ERROR\n", column);
			printf("%i: %f [CPU] vs %f [GPU] \n", i, cpuVal, gpuVal);
			exit(1);
		}
	}

}

void Tester::testColumnSolution(int* rowPtr, int* colIdx, double* A, double* b, double* x, int col,  const int dim)
{
	Eigen::MatrixXd cpuMatr = assemblyMatrixFromCSR(colIdx, rowPtr, NULL, dim, A);

	Eigen::VectorXd b_cpu = Eigen::VectorXd::Zero(dim);
//
	for (int j = 0; j < dim; j++) {
		b_cpu(j) =  b[j];
	}
//
	Eigen::VectorXd x_cpu = cpuMatr.colPivHouseholderQr().solve(b_cpu);

	printf("{SOLVING-RESULT} (for column %i) :\n", col);
//	std::cout << "b-Vector: \n" << b_cpu << std::endl;
//	std::cout << "x-Vector: \n" << x_cpu << std::endl;
//	std::cout << "Eigen Matrix: \n"<< cpuMatr << std::endl;


	printf("CULA\t\tEIGEN\n");
	for (int j = 0; j < dim; j++) {
		printf("%f\t\t%f\n", x[j], x_cpu(j));
	}


}


