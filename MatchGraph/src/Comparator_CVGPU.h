/*
 * Comparator_CVGPU.h
 *
 *  Created on: Jun 5, 2013
 *      Author: rodrigpro
 */

#ifndef COMPARATORCVGPU_H_
#define COMPARATORCVGPU_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp> //This is where actual SURF and SIFT algorithm is located
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <vector>

class ComparatorCVGPU {
public:
	int compareGPU(const char* img1, const char* img2, bool showMatches=true, bool drawEpipolar=false);
    
    int ratioTest(std::vector< std::vector<cv::DMatch> >& matches);
    
	void symmetryTest(const std::vector< std::vector<cv::DMatch> >& matches1,
                      const std::vector< std::vector<cv::DMatch> >& matches2,
                      std::vector<cv::DMatch>& symMatches);
    
    cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
                       const std::vector<cv::KeyPoint>& keypoints1,
                       const std::vector<cv::KeyPoint>& keypoints2,
                       std::vector<cv::DMatch>& outMatches);

	~ComparatorCVGPU(){};
};

#endif /* COMPARATORCVGPU_H_ */
