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

/** @function main */
int Comparator::compare(const char* img1, const char* img2)
{

	Mat im1 = imread( img1, CV_LOAD_IMAGE_COLOR );
	Mat im2 = imread( img2, CV_LOAD_IMAGE_COLOR );

	Mat im1r, im2r;
	CvSize size1 = cvSize(540, 540);
	CvSize size2 = cvSize(540, 540);
	//resize(im1, im1, size1);
	//resize(im2, im2, size2);
	//imshow("i1",im1);
	//imshow("i2",im2);

	//Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	//Mat img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

	//CV_LOAD_IMAGE_GRAYSCALE
	if( !im1.data || !im2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	float nndrRatio = 0.7f;

	//bool draw = argv[3]; // draw results?
	bool draw = false; // draw results?

	SurfFeatureDetector detector( minHessian );
	SurfDescriptorExtractor extractor;

	std::vector<KeyPoint> keypoints1, keypoints2;

	detector.detect( im1, keypoints1 );
	detector.detect( im2, keypoints2 );

	//-- Step 2: Calculate descriptors (feature vectors)
	//SiftDescriptorExtractor extractor(1, 3, 5.0, 5.0, 20.0);

	Mat descriptors1, descriptors2;

	extractor.compute( im1, keypoints1, descriptors1 );
	extractor.compute( im2, keypoints2, descriptors2 );


	//cout << "descriptors1 = " << descriptors1.size().height << endl << endl;
	//cout << "descriptors2 = "<< descriptors2.size().height << endl << endl;

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	//FlannBasedMatcher matcher;
	BFMatcher matcher;


	//std::vector< DMatch > matches;
	vector< vector< DMatch > > matches;
	//matcher.match( descriptors_object, descriptors_scene, good_matches );
	matcher.knnMatch( descriptors1, descriptors2, matches, 2);


	//	double max_dist = 0; double min_dist = 100;
	vector< DMatch > good_matches;
	good_matches.reserve(matches.size());

	for (size_t i = 0; i < matches.size(); ++i)
	{
		//if (matches[i].size() < 2)
		//            continue;

		const DMatch &m1 = matches[i][0];
		const DMatch &m2 = matches[i][1];

		if(m1.distance <= nndrRatio * m2.distance)
		{
			good_matches.push_back(m1);
		}
	}

	float k = (2 * good_matches.size()) / float(descriptors1.size().height + descriptors2.size().height);

	//cout << "k(I_i, I_j) = " << k << endl;

	//if (good_matches.size() < 10) return match;
	//if (good_matches.size() >= 10 && score >= 0.9) match = 1;


	//printf("-- #of matches : %zu \n", matches.size() );
	//printf("-- #of good matches : %zu \n", good_matches.size() );

	if (draw)
	{
		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

		std::vector< Point2f >  obj;
		std::vector< Point2f >  scene;

		for( unsigned int i = 0; i < good_matches.size(); i++ )
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
			scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
		}

		Mat img_matches;
		drawMatches( im1, keypoints1, im2, keypoints2,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object

		for( size_t i = 0; i < good_matches.size(); i++ )
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
			scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
		}

		Mat H = findHomography( obj, scene, CV_RANSAC );

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( im1.cols, 0 );
		obj_corners[2] = cvPoint( im1.cols, im1.rows ); obj_corners[3] = cvPoint( 0, im1.rows );
		std::vector<Point2f> scene_corners(4);

		perspectiveTransform( obj_corners, scene_corners, H);
		//cout << "H = "<< endl << " "  << H << endl << endl;

		//-- Draw lines between the corners (the mapped object in the image_1 - image_2 )
		line( img_matches, scene_corners[0] , scene_corners[1], Scalar(0, 255, 0), 2 ); //TOP line
		line( img_matches, scene_corners[1] , scene_corners[2], Scalar(0, 255, 0), 2 );
		line( img_matches, scene_corners[2] , scene_corners[3], Scalar(0, 255, 0), 2 );
		line( img_matches, scene_corners[3] , scene_corners[0] , Scalar(0, 255, 0), 2 );

		//-- Show detected matches
		//if (match)
		//{
		imshow( "Good Matches & Object detection", img_matches );
		waitKey(0);
		//}
	}
	if (k > 0.03) return 1;

	return -1;
}
