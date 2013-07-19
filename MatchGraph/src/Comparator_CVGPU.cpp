#include <stdio.h>
#include <iostream>
#include <stdlib.h>     /* abs */
#include <algorithm>    // std::max
#include "Comparator_CVGPU.h"

// Required to include CUDA vector types
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda.h>

// Thrust stuff
#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>


//Constructor
ComparatorCVGPU::ComparatorCVGPU()
{
	allowedOnGpu = 1000;
	onGpuCounter = 0;
	cv::gpu::DeviceInfo defInfo(0);
	double totalMem = (double) defInfo.totalMemory();
}

//Destructor
ComparatorCVGPU::~ComparatorCVGPU()
{
	//clean up map
	for(std::map<int, IMG*>::const_iterator iter = comparePairs.begin(); iter != comparePairs.end(); iter++)
	{
		iter->second->descriptors.release();
		iter->second->h_descriptors.release();
		delete iter->second;
	}
	comparePairs.clear();
}


int ComparatorCVGPU::compareGPU(ImageHandler* iHandler, int* h_idx1,int* h_idx2, int* h_result, int k, bool showMatches)
{
	if (getMemoryLoad() > 0.8f && onGpuCounter > 0)
	{
		printf("Begin of function. 80%%\n");
		cleanMap(NULL, 1);
	}

	cv::gpu::SURF_GPU surf;

	for (int i=0; i < k && h_idx1[i] < iHandler->getTotalNr(); i++) {
		IMG* i1;
		IMG* i2;
		if (getMemoryLoad() > 0.8f && onGpuCounter > 0)
		{
			printf("Begin of iteration. 80%%\n");
			cleanMap(NULL, 1);
		}
		std::map<int, IMG*>::const_iterator it1 = comparePairs.find(h_idx1[i]);
		if (it1 == comparePairs.end()) {

			//			if(onGpuCounter >= (allowedOnGpu - 1))
			//			if(percUsed > 0.8f && onGpuCounter > 0)
			//			{
			//				cleanMap(h_idx2[i]);
			//			}

			i1 = uploadImage(h_idx1[i], surf, iHandler);
			onGpuCounter++;
			comparePairs.insert(std::make_pair(h_idx1[i], i1));
			//						std::cout << "Picture inserted : " << h_idx1[i] << std::endl;
		}
		else
		{
			i1 = it1->second;
			if(!i1->gpuFlag)
			{ //not on gpu, have to (re-)upload
				//printf("Reuploading i1 [%i]", h_idx1[i]);
				//				if(percUsed > 0.8f && onGpuCounter > 1)
				//				{
				//					cleanMap(h_idx2[i]);
				//				}
				i1->descriptors.upload(i1->h_descriptors);
				i1->gpuFlag = true;
				onGpuCounter++;
			}
		}
		std::map<int, IMG*>::const_iterator it2 = comparePairs.find(h_idx2[i]);
		if (it2 == comparePairs.end()) {
			i2 = uploadImage(h_idx2[i], surf, iHandler);
			comparePairs.insert(std::make_pair(h_idx2[i], i2));

			//						std::cout << "Picture inserted : " << h_idx2[i] << std::endl;
		}
		else
		{
			i2 = it2->second;
			if(!i2->gpuFlag)
			{
				//printf("Reuploading i2 [%i]\n", h_idx2[i]);
				i2->descriptors.upload(i2->h_descriptors);
				i2->gpuFlag = true;
				onGpuCounter++;
			}
		}

		std::vector<cv::DMatch> symMatches;
		try {
			match2(i1->descriptors, i2->descriptors, symMatches);
		} catch (cv::Exception& e) {
			std::cout << "OpenCV exception!" << std::endl;
			e.msg;
			return 0;
		}

		float k = (2 * symMatches.size()) / float(i1->descriptors.size().height + i2->descriptors.size().height);
		//		std::cout << "i1.descriptors.size() " << i1.descriptors.size().height << endl;
		//		std::cout << "k(I_i, I_j) = " << k << std::endl;

		if (k < 0.04)
		{
			h_result[i] = 0;
		}
		else
		{
			h_result[i] = 1;
		}

		//clear vector
		//symMatches.clear();


		if (showMatches)
		{
			showPair(*i1, *i2, symMatches);

		}
		if (getMemoryLoad() > 0.8f && onGpuCounter > 0)
		{
			printf("End of iteration. 80%%\n");
			cleanMap(NULL, 1);
		}
	}

	surf.releaseMemory();
	return 1;
}

double ComparatorCVGPU::getMemoryLoad()
{
	double free = (double) defInfo.freeMemory();
	//printf("Free memory %.3f MB\n", free/1024/1024);
	//printf("Total memory %.3f MB\n", total/1024/1024);
	double percUsed = (totalMem-free)/totalMem;
	//printf("Percentage of used memory %.2f\n", percUsed);
	return percUsed;
}
void ComparatorCVGPU::cleanMap(int notAllowedI2, const int proportion)
{
	printf("++GOING TO CLEAN UP onGPU = %i, allowed = %i!\n", onGpuCounter, allowedOnGpu);

	double free = (double) defInfo.freeMemory();
	double percUsed = (totalMem-free)/totalMem;
	printf("Percentage of used memory %.2f\n", percUsed);
	
	int toRelease = onGpuCounter/proportion; //TODO how many?
	printf("Releasing %i\n", toRelease);
	for (std::map<int, IMG*>::iterator rmIter = comparePairs.begin();
			rmIter != comparePairs.end(); rmIter++) {
		if (rmIter->first != notAllowedI2) {
			IMG* current = rmIter->second;
			if (current->gpuFlag) {
				current->descriptors.download(current->h_descriptors); //download to host descriptor
				current->descriptors.release();
				current->gpuFlag = false;
				onGpuCounter--;
				toRelease--;
				if(toRelease == 0)
					break;
			}
		}
	}
}

//TODO: doesn't work yet!
void ComparatorCVGPU::showPair(IMG& img1, IMG& img2, std::vector<cv::DMatch>& symMatches)
{
	//	surf.downloadKeypoints(img1.keypoints, img1.h_keypoints);
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> points1, points2;
	cv::Mat im1 = cv::imread( img1.path, 0 );
	cv::Mat im2 = cv::imread( img2.path, 0 );

	// resize images
	//CvSize size1 = cvSize(540, 540);
	//CvSize size2 = cvSize(540, 540);
	//resize(im1, im1, size1);
	//resize(im2, im2, size2);

	for (std::vector<cv::DMatch>::const_iterator it = symMatches.begin();
			it != symMatches.end(); ++it) {

		// Get the position of left keypoints
		float x = img1.h_keypoints[it->queryIdx].pt.x;
		float y = img2.h_keypoints[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x,y));
		cv::circle(im1,cv::Point(x,y),3,cv::Scalar(255,0,0),2);
		// Get the position of right keypoints
		x = img2.h_keypoints[it->trainIdx].pt.x;
		y = img2.h_keypoints[it->trainIdx].pt.y;
		cv::circle(im2,cv::Point(x,y),3,cv::Scalar(255,0,0),2);
		points2.push_back(cv::Point2f(x,y));
	}

	cv::Mat img_matches;
	//cv::drawMatches(im1, im1_keypoints, im2, im2_keypoints, matches, img_matches);
	//cv::imshow("Matches", img_matches);

	cv::imshow("Image 1", im1);
	cv::imshow("Image 2", im2);

	cv::waitKey();
}

IMG* ComparatorCVGPU::uploadImage(const int inputImg, cv::gpu::SURF_GPU& surf, ImageHandler* iHandler) {
	struct IMG* img = new IMG();
	img->path = iHandler->getFullImagePath(inputImg);
	//	std::cout << img->path <<std::endl;
	cv::Mat imgfile = cv::imread( img->path, 0 );
	img->im_gpu.upload(imgfile);
	imgfile.release();  // release memory on the CPU
	try
	{
		surf(img->im_gpu, cv::gpu::GpuMat(), img->keypoints, img->descriptors, false);
	} catch (cv::Exception &Exception) {
		std::cout << "SURF OpenCV exception!" << std::endl;
		std::cout << img->path << std::endl;
		return NULL;
	}
	img->im_gpu.release(); // release memory on the GPU
	img->keypoints.release();

	img->gpuFlag = true;

	return img;
}
void ComparatorCVGPU::match2(cv::gpu::GpuMat& im1_descriptors_gpu, 
		cv::gpu::GpuMat& im2_descriptors_gpu,
		std::vector<cv::DMatch>& symMatches) {

	cv::gpu::BruteForceMatcher_GPU< cv::L2<float> > matcher;
	std::vector<std::vector< cv::DMatch> > matches1;
	std::vector<std::vector< cv::DMatch> > matches2;

	matcher.knnMatch(im1_descriptors_gpu, im2_descriptors_gpu, matches1, 2);
	matcher.knnMatch(im2_descriptors_gpu, im1_descriptors_gpu, matches2, 2);

	// 3. Remove matches for which NN ratio is > than threshold

	// clean image 1 -> image 2 matches
	ratioTest(matches1);
	// clean image 2 -> image 1 matches
	ratioTest(matches2);

	// 4. Remove non-symmetrical matches

	symmetryTest(matches1,matches2,symMatches);

	matcher.clear();
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
			if (((*matchIterator)[0].distance/(*matchIterator)[1].distance) > ratio) {

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

		if (matchIterator1->size() < 2) {// ignore deleted matches
			//if ((*matchIterator1)[1].queryIdx > 0) printf("NULL");
			continue;
		}

		// for all matches image 2 -> image 1
		for (std::vector< std::vector<cv::DMatch> >::const_iterator matchIterator2= matches2.begin();
				matchIterator2!= matches2.end(); ++matchIterator2) {

			if (matchIterator2->size() < 2) {// ignore deleted matches
				continue;
			}
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
