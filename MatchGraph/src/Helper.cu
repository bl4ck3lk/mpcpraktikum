/*
 * Helper.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: schwarzk
 */

#include "Helper.h"

int* Helper::downloadGPUArray(int* devPtr, int size)
{
    int* cpu = (int*) malloc(sizeof(int)*size);
    cudaMemcpy(cpu, devPtr, size*sizeof(int), cudaMemcpyDeviceToHost);
    return cpu;
}
