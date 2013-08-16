/*
 * Helper.h
 *
 * Header file for some static helper functions.
 *
 *  Created on: Jul 10, 2013
 *      Author: Armin, Fabian
 */

#ifndef HELPER_H_
#define HELPER_H_

#include <string>

class Helper {
public:
	Helper();
	virtual ~Helper();

	//Download a given memory device location to host and return the pointer.
	static int* downloadGPUArrayInt(int* devPtr, int size);
	static float* downloadGPUArrayFloat(float* devPtr, const int size);
	static double* downloadGPUArrayDouble(double* devPtr, const int size);

	//Wrap CUDA memcopy calls to use it in *.cpp files.
	static void cudaMemcpyArrayInt(int* h_src, int* d_trg, int size);
	static void cudaMemcpyArrayIntToHost(int* d_src, int* h_trg, int size);

	//Print device memory on console with a given message.
	static void printGpuArray(int* devPtr, const int size, const std::string message);
	static void printGpuArrayF(float* devPtr, const int size, const std::string message);
	static void printGpuArrayD(double * devPtr, const int size, std::string message);
	static void printGpuArrayL(long * devPtr, const int size, std::string message);
};

#endif /* HELPER_H_ */
