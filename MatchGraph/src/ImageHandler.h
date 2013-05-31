/*
 * ImageHandler.h
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#ifndef IMAGEHANDLER_H_
#define IMAGEHANDLER_H_

#include <map>

class ImageHandler
{
	public:
		ImageHandler(char* imgDir); //constructor
		//~ImageHandler(); //destructor
		char* getImage(int img); //get specific image
		int getTotalNr(); //get total nr of images
		int getMapSize();
		void printMap();
		void sortImages();

	private:
		std::map<int,char*> images;
		int nrImages; //total number of images
};

#endif /* CPUIMPL_H_ */
