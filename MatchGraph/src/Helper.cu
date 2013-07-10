/*
 * Helper.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: schwarzk
 */

#include "Helper.h"

int* Helper::downloadGPUArrayInt(int* devPtr, int size)
{
    int* cpu = (int*) malloc(sizeof(int)*size);
    cudaMemcpy(cpu, devPtr, size*sizeof(int), cudaMemcpyDeviceToHost);
    return cpu;
}

void Helper::cudaMemcpyArrayInt(int* h_src, int* d_trg, int size)
{
	cudaMemcpy(d_trg, h_src, size*sizeof(int), cudaMemcpyHostToDevice);
}
