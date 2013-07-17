/*
 * Helper.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: schwarzk
 */

#include "Helper.h"
#include "Tester.h"
#include <iostream>
#include <stdio.h>

#define CHECK_FOR_CUDA_ERROR 1

#define CUDA_CHECK_ERROR() {							\
cudaError_t err = cudaGetLastError();					\
if (cudaSuccess != err) {						\
    fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",	\
            __FILE__, __LINE__, cudaGetErrorString(err) );	\
    exit(EXIT_FAILURE);						\
}									\
}

void Helper::cudaMemcpyArrayInt(int* h_src, int* d_trg, int size)
{
	cudaMemcpy(d_trg, h_src, size*sizeof(int), cudaMemcpyHostToDevice);
}

void Helper::printGpuArray(int * devPtr, const int size, std::string message)
{
	int* cpu = (int*) malloc(sizeof(int)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(int), cudaMemcpyDeviceToHost);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	std::cout << message << " : ";
	Tester::printArrayInt(cpu, size);
	free(cpu);
}

void Helper::printGpuArrayF(float * devPtr, const int size, std::string message)
{
	float* cpu = (float*) malloc(sizeof(float)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(float), cudaMemcpyDeviceToHost);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	std::cout << message << " : ";
	Tester::printArrayFloat(cpu, size);
	free(cpu);
}

void Helper::printGpuArrayD(double * devPtr, const int size, std::string message)
{
	double* cpu = (double*) malloc(sizeof(double)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(double), cudaMemcpyDeviceToHost);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	std::cout << message << " : ";
	Tester::printArrayDouble(cpu, size);
	free(cpu);
}

void Helper::printGpuArrayL(long * devPtr, const int size, std::string message)
{
	long* cpu = (long*) malloc(sizeof(long)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(long), cudaMemcpyDeviceToHost);

#if CHECK_FOR_CUDA_ERROR
	CUDA_CHECK_ERROR()
#endif

	std::cout << message << " : ";
	Tester::printArrayLong(cpu, size);
	free(cpu);
}

int* Helper::downloadGPUArrayInt(int* devPtr, const int size)
{
	int* cpu = (int*) malloc(sizeof(int)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(int), cudaMemcpyDeviceToHost);
	return cpu;
}

float* Helper::downloadGPUArrayFloat(float* devPtr, const int size)
{
	float* cpu = (float*) malloc(sizeof(float)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(float), cudaMemcpyDeviceToHost);
	return cpu;
}

double* Helper::downloadGPUArrayDouble(double* devPtr, const int size)
{
	double* cpu = (double*) malloc(sizeof(double)*size);
	cudaMemcpy(cpu, devPtr, size*sizeof(double), cudaMemcpyDeviceToHost);
	return cpu;
}
