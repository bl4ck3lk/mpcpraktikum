/*
 * Tester.h
 *
 *  Created on: Jun 5, 2013
 *      Author: gufler
 */

#ifndef TESTER_H_
#define TESTER_H_

//template<typename T>
class Tester {
public:
	static void testLaplacian(char* gpuData, int* laplacian, int dim, float lambda);

	static void printArrayInt(int* arr, int n);
	static void printMatrixArrayInt(int* arr, int dim);

	static void printArrayChar(char* arr, int n);
	static void printMatrixArrayChar(char* arr, int dim);

	static void printArrayFloat(float* arr, int n);
};

#endif /* TESTER_H_ */
