/*
 * ImageHandler.cpp
 *
 * Initializes a given directory by indexing all images within this directory
 * in a map such that the image-path can be obtained fast by the corresponding
 * index.
 *
 *  Created on: 29.05.2013
 *      Author: furby
 */

#include "ImageHandler.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

//constructor
ImageHandler::ImageHandler(const char* imgDir)
	: nrImages(0), directory(imgDir)
{
	//construct map based on image directory and count images
    DIR           *dirHandle; 
    struct dirent *dirEntry;
    
    //open directory 
    dirHandle = opendir(imgDir); 
     
    if (dirHandle != NULL) 
    { 
        //get content 
        while ((dirEntry = readdir(dirHandle)) != NULL)
        { 
        	char* entry = (char*)dirEntry->d_name;
			if (strcmp(entry,".") != 0 && strcmp(entry,"..") != 0 && dirEntry->d_type == DT_REG)
			{
				images.insert(std::pair<int,std::string>(nrImages++, dirEntry->d_name));
			}
        }
        //done 
        closedir(dirHandle); 
    } 
}


//destructor
ImageHandler::~ImageHandler()
{
	images.clear();
}


//Get total number of images
int ImageHandler::getTotalNr()
{
	return nrImages;
}

int ImageHandler::getMapSize()
{
	return images.size();
}

void ImageHandler::printMap()
{
	for(std::map<int,std::string>::iterator it = images.begin(); it != images.end(); it++)
	{
		printf("[%i, %s]\n",it->first, it->second.c_str());
	}
}

//Get image path base on image nr
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

//Get full image path based on image nr.
const char* ImageHandler::getFullImagePath(int imgNr)
{
	std::string* dir = new std::string(directory);
	dir->append("/");
	return (dir->append(getImage(imgNr))).c_str();
}

void ImageHandler::fillWithEmptyImages(unsigned int num)
{
	images.clear();
	nrImages = 0;
	for(unsigned int i = 0; i < num; i++)
	{
		images.insert(std::pair<int,std::string>(nrImages++, ""));
	}
}
