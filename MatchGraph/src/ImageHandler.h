/*
 * ImageHandler.h
 *
 * Header file for the image handler. The image handler initializes and maps
 * the images of a given image directory to continuous numbers and provides
 * functions to retrieve the names and paths of the images.
 *
 *  Created on: 29.05.2013
 *      Author: Fabian
 */

#ifndef IMAGEHANDLER_H_
#define IMAGEHANDLER_H_

#include <map>
#include <string>

class ImageHandler
{
	public:
		ImageHandler(const char* imgDir, const char* imageExtension);
		~ImageHandler();

		//Get the image name for a given image number.
		const char* getImage(int img);

		//Get the entire image-path for a given image number.
		const char* getFullImagePath(int img);

		//Get the directory-path for which this image handler has been initialized.
		const std::string getDirectoryPath() const;

		//Get the total number of images.
		int getTotalNr();

		//Get the memory size of the image map.
		int getMapSize();

		//Print the entire image map on console.
		void printMap();

		//For testing and debugging. Fill the image map with dummy data.
		void fillWithEmptyImages(unsigned int num);

	private:
		//Image-Map.
		std::map<int,std::string> images;

		//Number of images in the image-map.
		int nrImages;

		//Directory-path on which this image handler has been initialized.
		std::string directory;
};

#endif /* CPUIMPL_H_ */
