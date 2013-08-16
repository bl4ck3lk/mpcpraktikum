/*
 * MatrixHandler.h
 *
 * Interface for the matrix representation of the match graph.
 *
 *  Created on: May 29, 2013
 *      Author: Armin, Fabian
 */

#ifndef MATRIXHANDLER_H_
#define MATRIXHANDLER_H_


class MatrixHandler {
public:
	virtual ~MatrixHandler(){};

	//Set a specific entry to true ('similar') or false ('dissimilar').
	virtual void set(int i, int j, bool val) = 0;

	//Return the size of the match graph.
	virtual unsigned int getDimension()= 0;

	//Return the confidence measure matrix F.
	virtual float* getConfMatrixF()= 0;

	//Return the current match graph as char array.
	virtual char* getMatrAsArray() = 0;

	//Return the status of a given image-pair.
	virtual char getVal(int i, int j) = 0;

	//Return the number of all image-pairs, that are marked as 'similar'.
	virtual int getSimilarities() = 0;

	//Print the matrix representation of the match graph on console.
	virtual void print() = 0;

	//Write the current match graph as GraphML file.
	virtual void writeGML(char* filename, bool similar, bool dissimilar, bool potential) = 0;
};

#endif /* MATRIXHANDLER_H_ */
