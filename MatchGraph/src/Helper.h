/*
 * Helper.h
 *
 *  Created on: Jul 10, 2013
 *      Author: schwarzk
 */

#ifndef HELPER_H_
#define HELPER_H_

class Helper {
public:
	Helper();
	virtual ~Helper();
	static int* downloadGPUArrayInt(int* devPtr, int size);
	static void cudaMemcpyArrayInt(int* h_src, int* d_trg, int size);
};

#endif /* HELPER_H_ */
