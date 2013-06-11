#include <cv.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>     /* abs */
#include <algorithm>    // std::max
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp> //This is where actual SURF and SIFT algorithm is located
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp> // for homography
#include <opencv2/imgproc/imgproc.hpp>
#include "Comparator.h"

#include <vector>

using namespace std;
using namespace cv;


int Comparator::compare(char* img1, char* img2, bool showMatches, bool drawEpipolar)
{

	Mat im1 = imread( img1, 0 );
	Mat im2 = imread( img2, 0 );

	Mat im1r, im2r;
	CvSize size1 = cvSize(im1.cols, 540);
	CvSize size2 = cvSize(im2.cols, 540);
	resize(im1, im1, size1);
	resize(im2, im2, size2);
	//imshow("i1",im1);
	//imshow("i2",im2);

	if( !im1.data || !im2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	//bool draw = argv[3]; // draw results?

//	SurfFeatureDetector detector(minHessian);
	SiftFeatureDetector detector;
	//SurfDescriptorExtractor extractor;

	std::vector<KeyPoint> keypoints1, keypoints2;

	detector.detect( im1, keypoints1 );
	detector.detect( im2, keypoints2 );

	//-- Step 2: Calculate descriptors (feature vectors)
//	SiftDescriptorExtractor extractor(1, 3, 5.0, 5.0, 20.0);
	SiftDescriptorExtractor extractor;

	Mat descriptors1, descriptors2;

	extractor.compute( im1, keypoints1, descriptors1 );
	extractor.compute( im2, keypoints2, descriptors2 );


	//cout << "descriptors1 = " << descriptors1.size().height << endl;
	//cout << "descriptors2 = "<< descriptors2.size().height << endl << endl;

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	//FlannBasedMatcher matcher;
	BFMatcher matcher(NORM_L2, false);

	std::vector<cv::DMatch> matches;
	// from image 1 to image 2
	// based on k nearest neighbours (with k=2)
	std::vector< std::vector<cv::DMatch> > matches1;
	matcher.knnMatch(descriptors1,descriptors2,
			matches1, // vector of matches (up to 2 per entry)
			2);		  // return 2 nearest neighbours

	// from image 2 to image 1
	// based on k nearest neighbours (with k=2)
	std::vector< std::vector<cv::DMatch> > matches2;
	matcher.knnMatch(descriptors2,descriptors1,
			matches2, // vector of matches (up to 2 per entry)
			2);		  // return 2 nearest neighbours

	std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;
	std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;

	// 3. Remove matches for which NN ratio is > than threshold

	// clean image 1 -> image 2 matches
	int removed = ratioTest(matches1);
	std::cout << "Number of matched points 1->2 (ratio test) : " << matches1.size()-removed << std::endl;
	// clean image 2 -> image 1 matches
	removed= ratioTest(matches2);
	std::cout << "Number of matched points 2->1 (ratio test) : " << matches2.size()-removed << std::endl;

	// 4. Remove non-symmetrical matches
	std::vector<cv::DMatch> symMatches;
	symmetryTest(matches1,matches2,symMatches);

	std::cout << "Number of matched points (symmetry test): " << symMatches.size() << std::endl;

	// 5. Validate matches using RANSAC
	cv::Mat fundamental= ransacTest(symMatches, keypoints1, keypoints2, matches);

	if (showMatches)
	{
		// draw the matches
		cv::Mat imageMatches;
		cv::drawMatches(im1,keypoints1,  // 1st image and its keypoints
				im2,keypoints2,  // 2nd image and its keypoints
				matches,			// the matches
				imageMatches,		// the image produced
				cv::Scalar(255,255,255)); // color of the lines
		cv::namedWindow("Matches");
		cv::imshow("Matches",imageMatches);
	}

	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;

	for (std::vector<cv::DMatch>::const_iterator it= symMatches.begin();
			it!= symMatches.end(); ++it) {

		// Get the position of left keypoints
		float x= keypoints1[it->queryIdx].pt.x;
		float y= keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x,y));
		cv::circle(im1,cv::Point(x,y),3,cv::Scalar(255,0,0),2);
		// Get the position of right keypoints
		x= keypoints2[it->trainIdx].pt.x;
		y= keypoints2[it->trainIdx].pt.y;
		cv::circle(im2,cv::Point(x,y),3,cv::Scalar(255,0,0),2);
		points2.push_back(cv::Point2f(x,y));
	}

	if (drawEpipolar) {
		// Draw the epipolar lines
		std::vector<cv::Vec3f> lines1;
		cv::computeCorrespondEpilines(cv::Mat(points1),1,fundamental,lines1);

		for (vector<cv::Vec3f>::const_iterator it= lines1.begin();
				it!=lines1.end(); ++it) {

			cv::line(im2,cv::Point(0,-(*it)[2]/(*it)[1]),
					cv::Point(im2.cols,-((*it)[2]+(*it)[0]*im2.cols)/(*it)[1]),
					cv::Scalar(255,255,255));
		}

		std::vector<cv::Vec3f> lines2;
		cv::computeCorrespondEpilines(cv::Mat(points2),2,fundamental,lines2);

		for (vector<cv::Vec3f>::const_iterator it= lines2.begin();
				it!=lines2.end(); ++it) {

			cv::line(im1,cv::Point(0,-(*it)[2]/(*it)[1]),
					cv::Point(im1.cols,-((*it)[2]+(*it)[0]*im1.cols)/(*it)[1]),
					cv::Scalar(255,255,255));
		}
	}

	// Display the images
	cv::namedWindow("Right Image Epilines (RANSAC)");
	cv::imshow("Right Image Epilines (RANSAC)",im1);
	cv::namedWindow("Left Image Epilines (RANSAC)");
	cv::imshow("Left Image Epilines (RANSAC)",im2);

	cv::waitKey();
	return -1;
}


int Comparator::ratioTest(std::vector< std::vector<cv::DMatch> >& matches) {

	int removed=0;
	float ratio = 0.85f;

	// for all matches
	for (std::vector< std::vector<cv::DMatch> >::iterator matchIterator = matches.begin();
			matchIterator!= matches.end(); ++matchIterator) {

		// if 2 NN has been identified
		if (matchIterator->size() > 1) {

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

// Insert symmetrical matches in symMatches vector
void Comparator::symmetryTest(const std::vector< std::vector<cv::DMatch> >& matches1,
		const std::vector< std::vector<cv::DMatch> >& matches2,
		std::vector<cv::DMatch>& symMatches) {

	// for all matches image 1 -> image 2
	for (std::vector< std::vector<cv::DMatch> >::const_iterator matchIterator1= matches1.begin();
			matchIterator1!= matches1.end(); ++matchIterator1) {

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
cv::Mat Comparator::ransacTest(const std::vector<cv::DMatch>& matches,
		const std::vector<cv::KeyPoint>& keypoints1,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::DMatch>& outMatches) {

	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
			it!= matches.end(); ++it) {

		// Get the position of left keypoints
		float x= keypoints1[it->queryIdx].pt.x;
		float y= keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x,y));
		// Get the position of right keypoints
		x= keypoints2[it->trainIdx].pt.x;
		y= keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x,y));
	}
	float confidence = 0.98;
	float distance = 3.0;
	bool refineF = false;
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(),0);
	cv::Mat fundemental = cv::findFundamentalMat(
			cv::Mat(points1),cv::Mat(points2), // matching points
			inliers,      // match status (inlier ou outlier)
			CV_FM_RANSAC, // RANSAC method
			distance,     // distance to epipolar line
			confidence);  // confidence probability

	// extract the surviving (inliers) matches
	std::vector<uchar>::const_iterator itIn= inliers.begin();
	std::vector<cv::DMatch>::const_iterator itM= matches.begin();
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

		for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();
				it!= outMatches.end(); ++it) {

			// Get the position of left keypoints
			float x= keypoints1[it->queryIdx].pt.x;
			float y= keypoints1[it->queryIdx].pt.y;
			points1.push_back(cv::Point2f(x,y));
			// Get the position of right keypoints
			x= keypoints2[it->trainIdx].pt.x;
			y= keypoints2[it->trainIdx].pt.y;
			points2.push_back(cv::Point2f(x,y));
		}

		// Compute 8-point F from all accepted matches
		fundemental= cv::findFundamentalMat(
				cv::Mat(points1),cv::Mat(points2), // matching points
				CV_FM_8POINT); // 8-point method
	}

	return fundemental;
}
/*
int main( int argc, char** argv )
{
	Comparator comp;
	int result = comp.compare("data/lib1.jpg", "data/lib3.jpg", false, false);
	cout << "result = " << result << endl;
}
*/
