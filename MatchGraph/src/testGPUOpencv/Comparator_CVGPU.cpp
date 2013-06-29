#include <cv.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>     /* abs */
#include <algorithm>    // std::max
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp> //This is where actual SURF and SIFT algorithm is located
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include "Comparator_CVGPU.h"

#include <vector>
// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

void ratio_aux(const int2 * trainIdx1, int2 * trainIdx2,
                            const float2 * distance1, const float2 * distance2);

struct IMG {
    cv::gpu::GpuMat im_gpu;
    cv::gpu::GpuMat keypoints, descriptors;
    std::vector<cv::KeyPoint> h_keypoints;
    std::vector<float> h_descriptors;
  };

// should be modified to: get an array of images, upload them all
// ? extract all features immediately?
// match a desired pair using the match2 function
int ComparatorCVGPU::compareGPU(std::string* images1, std::string* images2, int k, bool showMatches, bool drawEpipolar)
{
	std::map< std::string, IMG > comparePairs;

	SURF_GPU surf;
	for (int i=0; i < k; i++) {
		if (comparePairs.find(images1[i]) == comparePairs.end()) {
			comparePairs.insert(std::make_pair(images1[i], uploadImage(images1[i], surf)));
			std::cout << "Picture inserted : " << images1[i] << std::endl;
		}

		if (comparePairs.find(images2[i]) == comparePairs.end()) {
			comparePairs.insert(std::make_pair(images2[i], uploadImage(images2[i], surf)));
			std::cout << "Picture inserted : " << images2[i] << std::endl;
		}

		std::vector<cv::DMatch> symMatches;
		IMG& i1 = comparePairs.find(images1[i])->second;
		IMG& i2 = comparePairs.find(images2[i])->second;
		match2(i1.descriptors, i2.descriptors, symMatches);
	}

	// download result?
	// ransac??
	return 1;
}

IMG ComparatorCVGPU::uploadImage(const std::string& inputImg, SURF_GPU& surf) {
	struct IMG* img = new IMG();
	cv::Mat imgfile = imread( inputImg, 0 );
	img->im_gpu.upload(imgfile);
	surf(img->im_gpu, cv::gpu::GpuMat(), img->keypoints, img->descriptors, false);
	return *img;
}
void ComparatorCVGPU::match2(cv::gpu::GpuMat& im1_descriptors_gpu, 
				cv::gpu::GpuMat& im2_descriptors_gpu,
				std::vector<cv::DMatch>& symMatches) {

	cv::gpu::BruteForceMatcher_GPU< cv::L2<float> > matcher;
	//cv::gpu::GpuMat trainIdx, distance;
	std::vector<std::vector< cv::DMatch> > matches1;
	std::vector<std::vector< cv::DMatch> > matches2;
	
	matcher.knnMatch(im1_descriptors_gpu, im2_descriptors_gpu, matches1, 2);
	matcher.knnMatch(im2_descriptors_gpu, im1_descriptors_gpu, matches2, 2);

/*
	GpuMat trainIdxMat1, distanceMat1, allDist1;
	GpuMat trainIdxMat2, distanceMat2, allDist2;

	matcher.knnMatchSingle(im1_descriptors_gpu, im2_descriptors_gpu, 
				trainIdxMat1, distanceMat1, allDist1, 2);
	matcher.knnMatchSingle(im2_descriptors_gpu, im1_descriptors_gpu, 
				trainIdxMat2, distanceMat2, allDist2, 2);

	int2* trainIdx1 = trainIdxMat1.ptr<int2>();
	float2* distance1 = distanceMat1.ptr<float2>();
	int2* trainIdx2 = trainIdxMat2.ptr<int2>();
	float2* distance2 = distanceMat2.ptr<float2>();
	
    	std::cout << "Number of matched points 1->2 (trainIdxMat1):" << trainIdxMat1.size().width << std::endl;
	std::cout << "Number of matched points 1->2 (trainIdxMat2): " << trainIdxMat2.size().width << std::endl;

	std::cout << "Number of matched points 1->2 (matches1):" << matches1.size() << std::endl;
	std::cout << "Number of matched points 1->2 (matches2): " << matches2.size() << std::endl;
	//std::cout << "Number of matched points 2->1: " << allDist2.size << std::endl;
    // invokes a cuda kernel to process the matches (testing)
	ratio_aux(trainIdx1, trainIdx2,
              distance1, distance2);

	std::cout << "Kernel executed" << std::endl;
*/
	//TODO:idxs aus der GPU holen und testen, ob sie uebereinstimmen
/*
    for (unsigned int i = 0; i < im1_descriptors_gpu.size().width; ++i)
    	{
        //std::cout << "d " << p1 << std::endl;
        //reference[i].x = idata[i].x - idata[i].y;
        //reference[i].y = idata[i].y;
    	}
 */   
	//std::cout << "d " << trainIdx1[1].x << std::endl;
	
	//std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;
	//std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;

	// 3. Remove matches for which NN ratio is > than threshold
	std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;
	std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;
	// clean image 1 -> image 2 matches
	int removed = ratioTest(matches1);
	// clean image 2 -> image 1 matches
	removed = ratioTest(matches2);
	
    // 4. Remove non-symmetrical matches
	
	symmetryTest(matches1,matches2,symMatches);

	std::cout << "Number of matched points (symmetry test): " << symMatches.size() << std::endl;
}


//TODO: parallelize this method!
//this method iterates through the found matches and removes the matches for
//which the distance between the first and second match larger than 'ratio'
int ComparatorCVGPU::ratioTest(std::vector< std::vector<cv::DMatch> >& matches) {

	int removed = 0;
	float ratio = 0.85f;

	// for all matches
	for (std::vector< std::vector<cv::DMatch> >::iterator matchIterator = matches.begin();
			matchIterator!= matches.end(); ++matchIterator) {

		// if 2 NN has been identified
		if (matchIterator->size() > 1) {

			//TODO: parallelize it
			// check distance ratio
			if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio) {

				matchIterator->clear(); // remove match
				removed++;
			}
		} else { // does not have 2 neighbours

			matchIterator->clear(); // remove match
			removed++;
		}
	}

	return removed;
}

//TODO: parallelize this method!
// Insert symmetrical matches in symMatches vector
void ComparatorCVGPU::symmetryTest(const std::vector< std::vector<cv::DMatch> >& matches1,
		const std::vector< std::vector<cv::DMatch> >& matches2,
		std::vector<cv::DMatch>& symMatches) {

	// for all matches image 1 -> image 2
	for (std::vector< std::vector<cv::DMatch> >::const_iterator matchIterator1= matches1.begin();
			matchIterator1 != matches1.end(); ++matchIterator1) {

		if (matchIterator1->size() < 2) // ignore deleted matches
			continue;

		// for all matches image 2 -> image 1
		for (std::vector< std::vector<cv::DMatch> >::const_iterator matchIterator2= matches2.begin();
				matchIterator2!= matches2.end(); ++matchIterator2) {

			if (matchIterator2->size() < 2) // ignore deleted matches
				continue;

			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx  &&
					(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {

				// add symmetrical match
				symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
						(*matchIterator1)[0].trainIdx,
						(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

// Identify good matches using RANSAC
// Return fundemental matrix
cv::Mat ComparatorCVGPU::ransacTest(const std::vector<cv::DMatch>& matches,
		const std::vector<cv::KeyPoint>& keypoints1,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches) {

	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			it!= matches.end(); ++it) {

		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x,y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x,y));
	}
	float confidence = 0.98;
	float distance = 3.0;
	bool refineF = true;
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(),0);
	cv::Mat fundamental = cv::findFundamentalMat(
			cv::Mat(points1),cv::Mat(points2), // matching points
			inliers,      // match status (inlier ou outlier)
			CV_FM_RANSAC, // RANSAC method
			distance,     // distance to epipolar line
			confidence);  // confidence probability

	// extract the surviving (inliers) matches
	std::vector<uchar>::const_iterator itIn = inliers.begin();
	std::vector<cv::DMatch>::const_iterator itM = matches.begin();
	// for all matches
	for ( ;itIn!= inliers.end(); ++itIn, ++itM) {

		if (*itIn) { // it is a valid match

			outMatches.push_back(*itM);
		}
	}

	std::cout << "Number of matched points (after cleaning): " << outMatches.size() << std::endl;

	if (refineF) {
		// The F matrix will be recomputed with all accepted matches

		// Convert keypoints into Point2f for final F computation
		points1.clear();
		points2.clear();

		for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
				it != outMatches.end(); ++it) {

			// Get the position of left keypoints
			float x = keypoints1[it->queryIdx].pt.x;
			float y = keypoints1[it->queryIdx].pt.y;
			points1.push_back(cv::Point2f(x,y));
			// Get the position of right keypoints
			x = keypoints2[it->trainIdx].pt.x;
			y = keypoints2[it->trainIdx].pt.y;
			points2.push_back(cv::Point2f(x,y));
		}

		// Compute 8-point F from all accepted matches
		fundamental = cv::findFundamentalMat(
				cv::Mat(points1),cv::Mat(points2), // matching points
				CV_FM_8POINT); // 8-point method
	}

	return fundamental;
}

int main( int argc, char** argv )
{
	int k = 6;
	std::string images1[k];
	images1[0] = "paris1.jpg";
	images1[1] = "paris1.jpg";
	images1[2] = "House1.jpg";
	images1[3] = "House1.jpg";
	images1[4] = "castle1.jpg";
	images1[5] = "castle1.jpg";

	std::string images2[k];
	images2[0] = "paris2.jpg";
	images2[1] = "House1.jpg";
	images2[2] = "House2.jpg";
	images2[3] = "castle1.jpg";
	images2[4] = "castle2.jpg";
	images2[5] = "paris2.jpg";
	ComparatorCVGPU comp;
	int result = comp.compareGPU(images1, images2, k, true, false);
	//int result = comp.compareGPU(argv[1], argv[2], true, false);
	cout << "result = " << result << endl;
}
