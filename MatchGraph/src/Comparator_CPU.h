/*
 * Comparator.h
 *
 *  Created on: Jun 5, 2013
 *      Author: rodrigpro
 */

#ifndef COMPARATOR_CPU_H_
#define COMPARATOR_CPU_H_

class Comparator_CPU {
public:
	int compare(char* img1, char* img2, bool showMatches=false, bool drawEpipolar=false);

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