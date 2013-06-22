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

typedef std::map<int,std::set<int> > myElemMap;

class GPUSparse : public MatrixHandler{
private:
	const unsigned int dim;
	const float lambda;
	unsigned int num_similar; //# of similarities that is (number of 1s / 2)
	unsigned int num_dissimilar; //# of dissimilarities that is (number of -1s / 2)

	int* colIdx;
	int* rowPtr;
	int* degrees;
	int* diagPos; //position of diagonal elements within row

	myElemMap newElemMap;
	int numNewSimilar;
	int numNewDiagonal;
	
	myElemMap dissimilarMap;

	void addNewToRow(const int row, const int j);
	void addDissimilarToColumn(const int column, const int row);
	void incrementDegree(const int row);

public:
	GPUSparse();
	GPUSparse(unsigned int _dim, float _lambda);
	void updateSparseStatus(); //TODO later private?

	//~GPUMatrix();
	void set(int i, int j, bool val);
	unsigned int getDimension();
	float* getConfMatrixF();
	char* getMatrAsArray();
	char getVal(int i, int j);
	int getSimilarities();
	void print();
	void writeGML(char* filename, bool similar, bool dissimilar, bool potential);

	/* SPARSE-specific */
	float* getValueArr(bool gpuPointer, float* values, int* rowPtr, int* colIdx) const;
	float* getColumn(int i) const;
	int* getColIdx() const;
	int* getRowPtr() const;
	unsigned int getNNZ() const;

	static void prefixSumGPU(int* result, const int* array, const int dimension);
};

#endif /* GPUSPARSE_H_ */
