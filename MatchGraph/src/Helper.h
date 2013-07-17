/*
 * Helper.h
 *
 *  Created on: Jul 10, 2013
 *      Author: schwarzk
 */

#ifndef HELPER_H_
#define HELPER_H_

#include <string>

class Helper {
public:
	Helper();
	virtual ~Helper();
	static int* downloadGPUArrayInt(int* devPtr, int size);
	static float* downloadGPUArrayFloat(float* devPtr, const int size);
	static double* downloadGPUArrayDouble(double* devPtr, const int size);

	static void cudaMemcpyArrayInt(int* h_src, int* d_trg, int size);

	static void printGpuArray(int* devPtr, const int size, const std::string message);
	static void printGpuArrayF(float* devPtr, const int size, const std::string message);
	static void printGpuArrayD(double * devPtr, const int size, std::string message);
	static void printGpuArrayL(long * devPtr, const int size, std::string message);
};

#endif /* HELPER_H_ */
