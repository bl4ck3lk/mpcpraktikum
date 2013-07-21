#include <stdio.h>
#include <iostream>
#include "Comparator_CVGPU.h"


//Constructor
ComparatorCVGPU::ComparatorCVGPU()
{
	onGpuCounter = 0;
	totalMem = (double) devInfo.totalMemory();
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
	double memLoad;

	cv::gpu::SURF_GPU surf;

	for (int i=0; i < k && h_idx1[i] < iHandler->getTotalNr(); i++) {
		IMG* i1;
		IMG* i2;
		if ((memLoad = getMemoryLoad()) > 0.8 && onGpuCounter > 0)
		{
			//			printf("Begin of iteration.  Load = %f\n", memLoad);
			cleanMap(memLoad > .9 ? 1.0 : .5);
		}
		std::map<int, IMG*>::const_iterator it1 = comparePairs.find(h_idx1[i]);
		if (it1 == comparePairs.end()) {

			i1 = uploadImage(h_idx1[i], surf, iHandler);
			if(i1 == NULL)
			{ //uploading was not successful, e.g. bad file: do nothing, continue
				h_result[i] = 0;
				continue;
			}
			onGpuCounter++;
			comparePairs.insert(std::make_pair(h_idx1[i], i1));
			//						std::cout << "Picture inserted : " << h_idx1[i] << std::endl;
		}
		else
		{
			i1 = it1->second;
			if(!i1->gpuFlag)
			{ //not on gpu, have to (re-)upload
				i1->descriptors.upload(i1->h_descriptors);
				i1->gpuFlag = true;
				onGpuCounter++;
			}
		}
		std::map<int, IMG*>::const_iterator it2 = comparePairs.find(h_idx2[i]);
		if (it2 == comparePairs.end()) {
			i2 = uploadImage(h_idx2[i], surf, iHandler);
			if (i2 == NULL)
			{ //uploading was not successful, e.g. bad file: do nothing, continue
				h_result[i] = 0;
				continue;
			}
			comparePairs.insert(std::make_pair(h_idx2[i], i2));
			onGpuCounter++;
			//						std::cout << "Picture inserted : " << h_idx2[i] << std::endl;
		}
		else
		{
			i2 = it2->second;
			if(!i2->gpuFlag)
			{
				i2->descriptors.upload(i2->h_descriptors);
				i2->gpuFlag = true;
				onGpuCounter++;
			}
		}

		std::vector<cv::DMatch> symMatches;
		try {
			match2(i1->descriptors, i2->descriptors, symMatches);
			float k = (2 * symMatches.size()) / float(i1->descriptors.size().height + i2->descriptors.size().height);
			//		std::cout << "i1.descriptors.size() " << i1.descriptors.size().height << endl;
			//			std::cout << "k(I_i, I_j) = " << k << std::endl;

			if (k < 0.025)
			{
				h_result[i] = 0;
			}
			else
			{
				h_result[i] = 1;
			}

			if (showMatches)
			{
				showPair(*i1, *i2, symMatches, surf);

			}
		} catch (cv::Exception& e) {
			std::cout << "OpenCV exception!" << std::endl;
			h_result[i] = 0;
		}
	}

	surf.releaseMemory();
	return 1;
}

double ComparatorCVGPU::getMemoryLoad()
{
	const double free = (double) devInfo.freeMemory();
	//	printf("Free memory %.3f MB (of total %.3f)\n", free/1024/1024, totalMem/1024/1024);

	return (totalMem-free)/totalMem;
}
void ComparatorCVGPU::cleanMap(const float proportion)
{
	int toRelease = onGpuCounter/proportion;

	//double free = (double) defInfo.freeMemory();
	//double percUsed = (totalMem-free)/totalMem;
	//	printf("CLEAN UP: Percentage of used memory %.2f -> releasing: %i\n", percUsed, toRelease);

	for (std::map<int, IMG*>::const_iterator rmIter = comparePairs.begin();
			rmIter != comparePairs.end(); rmIter++)
	{

		IMG* current = rmIter->second;
		if (current->gpuFlag)
		{
			current->descriptors.download(current->h_descriptors); //download to host descriptor
			current->descriptors.release();
			current->gpuFlag = false;
			onGpuCounter--;
			toRelease--;
			if (toRelease == 0)
				break;
		}

	}
}

//TODO: doesn"t work if keypoints are released
void ComparatorCVGPU::showPair(IMG& img1, IMG& img2, std::vector<cv::DMatch>& symMatches, cv::gpu::SURF_GPU& surf)
{
	surf.downloadKeypoints(img1.keypoints, img1.h_keypoints);
	surf.downloadKeypoints(img2.keypoints, img2.h_keypoints);

	cv::Mat im1 = cv::imread( img1.path, 0 );
	cv::Mat im2 = cv::imread( img2.path, 0 );

	// resize images
	//CvSize size1 = cvSize(540, 540);
	//CvSize size2 = cvSize(540, 540);
	//resize(im1, im1, size1);
	//resize(im2, im2, size2);

	cv::Mat img_matches;
	cv::drawMatches(im1, img1.h_keypoints, im2, img2.h_keypoints, symMatches, img_matches,
			cv::Scalar::all(-1), cv::Scalar::all(-1),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	cv::imshow("Matches", img_matches);
	cv::waitKey();

	//TODO: path to save
//	if (write)
//	{
//		std::string id1str = static_cast<std::ostringstream*>( &(std::ostringstream() << id1) )->str();
//		std::string id2str = static_cast<std::ostringstream*>( &(std::ostringstream() << id2) )->str();
//		cv::imwrite(id1str+"_"+id2str+".jpg", img_matches);
//	}
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
		img->im_gpu.release();
		img->keypoints.release();
		delete img;
		return NULL;
	}
	img->im_gpu.release(); // release memory on the GPU
	img->keypoints.release(); //TODO: don't release if in showMatches modus

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
	const float ratio = 0.85f;

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
