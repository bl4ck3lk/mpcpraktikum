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
#include "Comparator_CVGPU.h"

#include <vector>

using namespace std;
using namespace cv;
using namespace cv::gpu;


int ComparatorCVGPU::compareGPU(char* img1, char* img2, bool showMatches, bool drawEpipolar)
{
	cv::Mat im1 = imread( img1, 0 );
	cv::Mat im2 = imread( img2, 0 );

	//Mat im1r, im2r;
	//CvSize size1 = cvSize(im1.cols, 540);
	//CvSize size2 = cvSize(im2.cols, 540);
	//resize(im1, im1, size1);
	//resize(im2, im2, size2);
	//imshow("i1",im1);
	//imshow("i2",im2);

	if( !im1.data || !im2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	cv::gpu::GpuMat im1_gpu, im2_gpu;
	cv::gpu::GpuMat im1_keypoints_gpu, im1_descriptors_gpu;
	cv::gpu::GpuMat im2_keypoints_gpu, im2_descriptors_gpu;
	SURF_GPU surf;
 
	std::vector<cv::KeyPoint> im1_keypoints, im2_keypoints;
	std::vector<float> im1_descriptors, im2_descriptors;
	std::vector<std::vector< cv::DMatch> > matches;
 
		// upload images into the GPU
		im1_gpu.upload(im1);
		im2_gpu.upload(im2);
 
		// detect keypoints & compute descriptors
		surf(im1_gpu, cv::gpu::GpuMat(), im1_keypoints_gpu, im1_descriptors_gpu, false);
		surf(im2_gpu, cv::gpu::GpuMat(), im2_keypoints_gpu, im2_descriptors_gpu, false);
 

		surf.downloadKeypoints(im1_keypoints_gpu, im1_keypoints);
		surf.downloadKeypoints(im2_keypoints_gpu, im2_keypoints);
		surf.downloadDescriptors(im1_descriptors_gpu, im1_descriptors);
		surf.downloadDescriptors(im2_descriptors_gpu, im2_descriptors);
 

		cv::gpu::BruteForceMatcher_GPU< cv::L2<float> > matcher;
		cv::gpu::GpuMat trainIdx, distance;
		matcher.radiusMatch(im1_descriptors_gpu, im2_descriptors_gpu, matches, 0.1f);
 
		Mat img_matches;
		if (showMatches) {
		cv::drawMatches(im1, im1_keypoints, im2, im2_keypoints, matches, img_matches);
 
		cv::imshow("Matches", img_matches);
		cv::waitKey();
	 	}
 
return 0;

}

/*
int main( int argc, char** argv )
{
	ComparatorCVGPU comp;
	int result = comp.compareGPU(argv[1], argv[2], true, false);
	cout << "result = " << result << endl;
}
*/
