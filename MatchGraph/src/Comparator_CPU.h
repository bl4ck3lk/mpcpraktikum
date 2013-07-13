/*
 * Comparator_CPU.h
 *
 *  Created on: Jun 5, 2013
 *      Author: rodrigpro
 */

#ifndef COMPARATOR_CPU_H_
#define COMPARATOR_CPU_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp> //This is where actual SURF and SIFT algorithm is located
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

class Comparator_CPU {
public:
	int compare(const char* img1, const char* img2, bool showMatches=false, bool drawEpipolar=false);

	int ratioTest(std::vector< std::vector<cv::DMatch> >& matches);

	void symmetryTest(const std::vector< std::vector<cv::DMatch> >& matches1,
			const std::vector< std::vector<cv::DMatch> >& matches2,
			std::vector<cv::DMatch>& symMatches);

	 cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,
			                 const std::vector<cv::KeyPoint>& keypoints1,
							 const std::vector<cv::KeyPoint>& keypoints2,
						     std::vector<cv::DMatch>& outMatches);
	~Comparator_CPU(){};
};

#endif /* COMPARATOR_CPU_H_ */
