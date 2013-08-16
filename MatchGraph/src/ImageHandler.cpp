/*
 * ImageHandler.cpp
 *
 * Initializes a given directory by indexing all images within this directory
 * in a map such that the image-path can be obtained fast by the corresponding
 * index.
 *
 *  Created on: 29.05.2013
 *      Author: Fabian
 */

#include "ImageHandler.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

/*
 * Constructor
 */
ImageHandler::ImageHandler(const char* imgDir, const char* imageExtension)
	: nrImages(0), directory(imgDir)
{
	//construct map based on image directory and count images
    DIR           *dirHandle; 
    struct dirent *dirEntry;
    
    //open directory 
    dirHandle = opendir(imgDir); 

    //extension length
    int extensionLength = strlen(imageExtension);
     
    if (dirHandle != NULL) 
    { 
        //get content 
        while ((dirEntry = readdir(dirHandle)) != NULL)
        { 
        	char* entry = (char*)dirEntry->d_name;

        	//ensure imageExtension is at the end of the file name
        	int entryLength = strlen(entry);

        	//file filter, only initialize images with the given file extension
			if (strcmp(entry,".") != 0 && strcmp(entry,"..") != 0 && strstr(entry + (entryLength-extensionLength), imageExtension) != NULL && dirEntry->d_type == DT_REG)
			{
				images.insert(std::pair<int,std::string>(nrImages++, entry));
			}
        }
        //done 
        closedir(dirHandle); 
    } 
}


/*
 * Destructor
 */
ImageHandler::~ImageHandler()
{
	images.clear();
}


/*
 * Return the total number of initialized images.
 */
int ImageHandler::getTotalNr()
{
	return nrImages;
}

/*
 * Return the memory size of the image map.
 */
int ImageHandler::getMapSize()
{
	return images.size();
}

/*
 * Print the entire image-map.
 */
void ImageHandler::printMap()
{
	for(std::map<int,std::string>::iterator it = images.begin(); it != images.end(); it++)
	{
		printf("[%i, %s]\n",it->first, it->second.c_str());
	}
}

/*
 * Return the image name for a given image number.
 */
const char* ImageHandler::getImage(int imgNr)
{
	if (imgNr < 0 || imgNr > nrImages-1) //images: 0,....,nrImages-1
	{
		printf("ImageHandler Error: Image nr %i not within [0,%i].\nReturning image 0.",imgNr,nrImages-1);
		return images.find(0)->second.c_str();
	} else {
		return images.find(imgNr)->second.c_str();
	}

}

/*
 * Return the full image-path for the given image number.
 */
const char* ImageHandler::getFullImagePath(int imgNr)
{
	std::string* dir = new std::string(directory);
	dir->append("/");
	return (dir->append(getImage(imgNr))).c_str();
}

/*
 * Function for debugging and testing.
 * Fill the image map with a given number of dummy data.
 */
void ImageHandler::fillWithEmptyImages(unsigned int num)
{
	images.clear();
	nrImages = 0;
	for(unsigned int i = 0; i < num; i++)
	{
		images.insert(std::pair<int,std::string>(nrImages++, ""));
	}
}

/*
 * Return the directory-path for which this image handler has been initialized.
 */
const std::string ImageHandler::getDirectoryPath() const
{
	return directory;
}
