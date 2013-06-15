/*
 * ImageHandler.h
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#ifndef IMAGEHANDLER_H_
#define IMAGEHANDLER_H_

#include <map>
#include <string>

class ImageHandler
{
	public:
		ImageHandler(const char* imgDir); //constructor
		//~ImageHandler(); //destructor
		const char* getImage(int img); //get specific image
		const char* getFullImagePath(int img);
		int getTotalNr(); //get total nr of images
		int getMapSize();
		void printMap();
		void sortImages();
		void fillWithEmptyImages(unsigned int num);

	private:
		std::map<int,std::string> images;
		int nrImages; //total number of images
		std::string directory;
};

#endif /* CPUIMPL_H_ */
