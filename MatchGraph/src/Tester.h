/*
 * Tester.h
 *
 *  Created on: Jun 5, 2013
 *      Author: gufler
 */

#ifndef TESTER_H_
#define TESTER_H_

class Tester {
public:
	static void testLaplacian(char* gpuData, int* laplacian, int dim, float lambda);
};

#endif /* TESTER_H_ */
