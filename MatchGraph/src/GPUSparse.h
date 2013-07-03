/*
 * GPUSparse.h
 *
 *  Created on: Jun 12, 2013
 *      Author: gufler
 */

#ifndef GPUSPARSE_H_
#define GPUSPARSE_H_

#include "MatrixHandler.h"
#include <set>
#include <map>
#include <string>
#include <boost/unordered_map.hpp> 
#include <boost/unordered_set.hpp> 

typedef std::map<int,std::set<int> > myElemMap;
typedef boost::unordered_map<int, boost::unordered_set<int> > IndexMap;

class GPUSparse : public MatrixHandler{
private:
	const unsigned int dim;
	const float lambda;
	unsigned int num_similar; //# of similarities that is (number of 1s / 2)
	unsigned int num_dissimilar; //# of dissimilarities that is (number of -1s / 2)
	unsigned int nnz_rows; // # of rows having non-zero elements

	bool firstInitMode;

	int* _gpuDiagPos; //position of diagonal elements within row
	int* _gpuDegrees;
	int* _gpuRowPtr;
	int* _gpuColIdx;
	
	myElemMap dissimilarMap;
	
	IndexMap similarMap;

	void addNewToRow(const int row, const int j);
	void addDissimilarToColumn(const int column, const int row);
	void incrementDegree(const int row);

public:
	GPUSparse();
	GPUSparse(unsigned int _dim, float _lambda);
	void updateSparseStatus(int* _idx1, int* _idx2, int* _res, int _k);
	void handleDissimilar(int* idxData, int num);

	~GPUSparse();
	void set(int i, int j, bool val);
	unsigned int getDimension();
	float* getConfMatrixF();
	char* getMatrAsArray();
	char getVal(int i, int j);
	int getSimilarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);

	/* SPARSE-specific */
	float* getValueArr(bool gpuPointer) const;
	float* getColumn(int i) const;
	int* getColIdx() const;
	int* getColIdxDevice() const;
	int* getRowPtr() const;
	int* getRowPtrDevice() const;
	unsigned int getNNZ() const;
	void setRandom(int num);

	static int* prefixSumGPU(int* result, const int* array, const int dimension);
	static void printGpuArray(int* devPtr, const int size, const std::string message);
	static void printGpuArrayF(float* devPtr, const int size, const std::string message);
};

#endif /* GPUSPARSE_H_ */

