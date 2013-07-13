/*
 * Tester.h
 *
 *  Created on: Jun 5, 2013
 *      Author: gufler
 */

#ifndef TESTER_H_
#define TESTER_H_

#include <set>
#include <map>

//template<typename T>
class Tester {
public:
	static void testLaplacian(char* gpuData, int* laplacian, int dim, float lambda);

	static void printMatrixArrayChar(char* arr, int dim);
	static void printMatrixArrayInt(int* arr, int dim);

	template<typename T>
	static void printArray(T* array, const int size);
	//CUDA seems not to like template
	static void printArrayInt(int* arr, int n);
	static void printArrayChar(char* arr, int n);
	static void printArrayFloat(float* arr, int n);
	static void printArrayLong(long* arr, int n);
	static void printArrayDouble(double* arr, int n);

	static void testCSRMatrixUpdate(int* origRowPtr, int* origColIdx, int* degrees,
			int* newRowPtr, int* newColIdx, int* idx1, int* idx2, int* res,  std::map<int,std::set<int> > dissimilarMap, int dim, int k);

	static void testValueArray(int* rowPtr, int* colIdx, int* degrees, int dim, int nnz, int lambda, double* gpuResult);

	static void testColumn(std::map<int,std::set<int> > dissimilarMap, int* rowPtr, int* colIdx, int column, int dim, double* gpuResult);

	static void testColumnSolution(int* rowPtr, int* colIdx, double* A, double* b, double* x, int col,  const int dim);
};

#endif /* TESTER_H_ */
