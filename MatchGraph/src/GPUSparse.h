/*
 * GPUSparse.h
 *
 *  Created on: Jun 12, 2013
 *      Author: gufler
 */

#ifndef GPUSPARSE_H_
#define GPUSPARSE_H_

#include "MatrixHandler.h"
#include "ImageHandler.h"
#include <set>
#include <map>
#include <string>


typedef std::map<int,std::set<int> > myElemMap;

class GPUSparse : public MatrixHandler{
private:
	// the dimension of the matrix (symmetric matrix)
	const unsigned int dim;

	// algorithm parameter lambda
	const float lambda;

	//# of similarities that is (number of 1s / 2)
	unsigned int num_similar;

	//position of diagonal elements within row
	int* _gpuDiagPos;

	//degree of diagonal elements (neighbors of a node)
	int* _gpuDegrees;

	// the row pointer array (CSR sparse matrix format)
	int* _gpuRowPtr;

	//the column index array (CSR sparse matrix format)
	int* _gpuColIdx;
	
	//a map containing all dissimilar elements with the row index as key and a set
	//of column indices as value
	myElemMap dissimilarMap;

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

	/* SPARSE-specific functions (not in virtual superclass) */
	unsigned int getNNZ() const;
	int* getColIdxDevice() const;
	int* getRowPtrDevice() const;

	float* getValueArray(bool gpuPointer) const;
	double* getValueArrayDouble(bool gpuPointer) const;

	float* getColumn(int i) const;
	double* getColumnDouble(int i) const;

	void logSimilarToFile(const char *path, ImageHandler* iHandler) const;
};

#endif /* GPUSPARSE_H_ */

