/*
 * Comparator_CVGPU.h
 *
 *  Created on: Jun 5, 2013
 *      Author: rodrigpro
 */

#ifndef COMPARATORCVGPU_H_
#define COMPARATORCVGPU_H_

class ComparatorCVGPU {
public:
	int compareGPU(char* img1, char* img2, bool showMatches=true, bool drawEpipolar=false);

	~ComparatorCVGPU(){};
};

#endif /* COMPARATORCVGPU_H_ */
