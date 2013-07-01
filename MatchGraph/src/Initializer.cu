/*
 * Initializer.cu
 *
 * Initializes a given directory by indexing all images within this directory
 * in a map such that the image-path can be obtained fast by the corresponding
 * index.
 * Initializes the T Matrix.
 *
 *  Created on: Jun 29, 2013
 *      Author: schwarzk
 */

#include "Initializer.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

Initializer::Initializer(const char* imgDir)
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

Initializer::~Initializer()
{
	images.clear();
}

//Get total number of images
int Initializer::getTotalNr()
{
	return nrImages;
}

int Initializer::getMapSize()
{
	return images.size();
}

void Initializer::printMap()
{
	for(std::map<int,std::string>::iterator it = images.begin(); it != images.end(); it++)
	{
		printf("[%i, %s]\n",it->first, it->second.c_str());
	}
}

//Get image path base on image nr
const char* Initializer::getImage(int imgNr)
{
	if (imgNr < 0 || imgNr > nrImages-1) //images: 0,....,nrImages-1
	{
		printf("Initializer Error: Image nr %i not within [0,%i].\nReturning image 0.",imgNr,nrImages-1);
		return images.find(0)->second.c_str();
	} else {
		return images.find(imgNr)->second.c_str();
	}

}

//Get full image path based on image nr.
const char* Initializer::getFullImagePath(int imgNr)
{
	std::string* dir = new std::string(directory);
	dir->append("/");
	return (dir->append(getImage(imgNr))).c_str();
}

void Initializer::fillWithEmptyImages(unsigned int num)
{
	images.clear();
	nrImages = 0;
	for(unsigned int i = 0; i < num; i++)
	{
		images.insert(std::pair<int,std::string>(nrImages++, ""));
	}
}

/*
 * Initializes the T-Matrix such that, similarThreshold similarities are set or maxUpdateArraySize is reached.
 * Comparator is invoked with arrays of size comparatorArraySize.
 * Returns true, if similarThreshold is reached, false if maxUpdateArraySize is reached.
 */
bool Initializer::doInitializationPhase(MatrixHandler* T, int comparatorArraySize, int similarThreshold, int maxUpdateArraySize)
{

	/* Vorgehensweise:
	 * 1. Array mit random-longs (kontinuierlicher Index) auf der GPU erzeugen. (cuRand)
	 * 2. Kernel schaut sich jeden Eintrag an und pr[ft ob innerhalb der Diagonalmatrix
	 * 		und noch keine Infomation ueber Eintrag. Falls doch wird Eintrag auf -1 gesetzt.
	 * 3. Array absteigend sortieren (thrust) [...... ; -1, -1 ...].
	 * 4. Kernel schreibt die nicht -1 Indices in die 'richtigen' Index-Arrays.
	 * 5. Array sortieren.
	 */

	return true;
}

