#include <cv.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>     /* abs */
#include <algorithm>    // std::max
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

// should be modified to: get an array of images, upload them all
// ? extract all features immediately?
// match a desired pair using the match2 function
int ComparatorCVGPU::compareGPU(char* img1, char* img2, bool showMatches, bool drawEpipolar)
{
	// TODO: images should be passed as an array of images pairs.
	cv::Mat im1 = imread( img1, 0 );
	cv::Mat im2 = imread( img2, 0 );

	//Mat im1r, im2r;
	//CvSize size1 = cvSize(im1.cols, 540);
	//CvSize size2 = cvSize(im2.cols, 540);
	//resize(im1, im1, size1);
	//resize(im2, im2, size2);
	//imshow("i1",im1);
	//imshow("i2",im2);

	// for each image pair in input array do:
	if( !im1.data || !im2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	// use new ...
	cv::gpu::GpuMat im1_gpu, im2_gpu;
	cv::gpu::GpuMat im1_keypoints_gpu, im1_descriptors_gpu;
	cv::gpu::GpuMat im2_keypoints_gpu, im2_descriptors_gpu;

	std::vector<std::vector < cv::gpu::GpuMat*> > pairs;
	std::vector < cv::gpu::GpuMat*> tmp;
	
	tmp.push_back(&im1_gpu);
	tmp.push_back(&im2_gpu);
	tmp.push_back(&im1_keypoints_gpu);
	tmp.push_back(&im2_keypoints_gpu);

	tmp.push_back(&im1_descriptors_gpu);
	tmp.push_back(&im2_descriptors_gpu);

	pairs.push_back(tmp);
	SURF_GPU surf;
 
	std::vector<cv::KeyPoint> im1_keypoints, im2_keypoints;
	std::vector<float> im1_descriptors, im2_descriptors;
 	// end for

	// upload all the necessary data for each comp-pair
	for (std::vector<std::vector < cv::gpu::GpuMat*> >::iterator it = pairs.begin();
it != pairs.end(); ++it) {
		//it[0] - im1
		//it[1] - im2
		(*it)[0]->upload(im1); 
		(*it)[1]->upload(im2);
		// im1 and im2 above should come from another vector/array with
		// correspondent indices
		// upload should be done only if not done yet!
		// once done, save this information!


		// detect keypoints & compute descriptors
		surf(*(*it)[0], cv::gpu::GpuMat(), *(*it)[2], *(*it)[4], false);
		surf(*(*it)[1], cv::gpu::GpuMat(), *(*it)[3], *(*it)[5], false);
		// release im1 and im2	
	}
	// upload images into the GPU
	//im1_gpu.upload(im1);
	//im2_gpu.upload(im2);
	/*	
	if (!showMatches) // free memory we won't use again
	{
 		im1.release();
		im2.release();
	}
	// detect keypoints & compute descriptors
	surf(im1_gpu, cv::gpu::GpuMat(), im1_keypoints_gpu, im1_descriptors_gpu, false);
	surf(im2_gpu, cv::gpu::GpuMat(), im2_keypoints_gpu, im2_descriptors_gpu, false);
 	*/
	std::vector<cv::DMatch> symMatches;
	// compare two images using their gpu_descriptors
	// test match only for pairs in the 'TOMATCH-list'
	match2(im1_descriptors_gpu, im2_descriptors_gpu, symMatches);

	surf.downloadKeypoints(im1_keypoints_gpu, im1_keypoints);
	surf.downloadKeypoints(im2_keypoints_gpu, im2_keypoints);
	surf.downloadDescriptors(im1_descriptors_gpu, im1_descriptors);
	surf.downloadDescriptors(im2_descriptors_gpu, im2_descriptors);

	// 5. Validate matches (clean more) using RANSAC
	//std::vector<cv::DMatch> matches;
	//cv::Mat fundamental = ransacTest(symMatches, im1_keypoints, im2_keypoints, matches);


	//if (symMatches.size() < thresholdMatchPoints) return -1;

	//TODO: not quite correct. It should be normalized by descriptors size.
	float k = (2 * symMatches.size()) / float(im1_keypoints.size() + im2_keypoints.size());
	cout << "k(I_i, I_j) = " << k << endl;
	
	//if (k < 0.01) return -1;
	
	if (showMatches) 
	{	
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;

	for (std::vector<cv::DMatch>::const_iterator it = symMatches.begin();
			it != symMatches.end(); ++it) {

		// Get the position of left keypoints
		float x = im1_keypoints[it->queryIdx].pt.x;
		float y = im1_keypoints[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x,y));
		cv::circle(im1,cv::Point(x,y),3,cv::Scalar(255,0,0),2);
		// Get the position of right keypoints
		x = im2_keypoints[it->trainIdx].pt.x;
		y = im2_keypoints[it->trainIdx].pt.y;
		cv::circle(im2,cv::Point(x,y),3,cv::Scalar(255,0,0),2);
		points2.push_back(cv::Point2f(x,y));
	}

		Mat img_matches;
		//cv::drawMatches(im1, im1_keypoints, im2, im2_keypoints, matches, img_matches);
		//cv::imshow("Matches", img_matches);
		
		cv::imshow("Image 1", im1);
		cv::imshow("Image 2", im2);
		
		cv::waitKey();
	 }
 
	return 1;

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
	ComparatorCVGPU comp;
	int result = comp.compareGPU(argv[1], argv[2], true, false);
	cout << "result = " << result << endl;
}
